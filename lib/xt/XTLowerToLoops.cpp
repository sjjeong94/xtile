#include "xt/XTPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::xt;

namespace mlir::xt {
#define GEN_PASS_DEF_XTLOWERTOLOOPS
#include "xt/XTPasses.h.inc"
} // namespace mlir::xt

static Value createIndexConstant(PatternRewriter &rewriter, Location loc,
                                 int64_t value) {
  return rewriter.create<arith::ConstantIndexOp>(loc, value);
}

static Value asIndex(PatternRewriter &rewriter, Location loc, Value value) {
  if (value.getType().isIndex())
    return value;
  return rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), value);
}

static Value createTensorLoopNest(
    PatternRewriter &rewriter, Location loc, RankedTensorType tensorType,
    function_ref<Value(OpBuilder &, Location, ValueRange)> bodyBuilder) {
  auto shape = tensorType.getShape();
  Value empty = tensor::EmptyOp::create(rewriter, loc, shape,
                                        tensorType.getElementType());
  SmallVector<Value> lbs, ubs, steps;
  for (int64_t dim : shape) {
    lbs.push_back(createIndexConstant(rewriter, loc, 0));
    ubs.push_back(createIndexConstant(rewriter, loc, dim));
    steps.push_back(createIndexConstant(rewriter, loc, 1));
  }
  scf::LoopNest loopNest = scf::buildLoopNest(
      rewriter, loc, lbs, ubs, steps, ValueRange{empty},
      [&](OpBuilder &builder, Location nestedLoc, ValueRange ivs,
          ValueRange iterArgs) -> scf::ValueVector {
        Value elem = bodyBuilder(builder, nestedLoc, ivs);
        Value next =
            builder.create<tensor::InsertOp>(nestedLoc, elem, iterArgs[0], ivs);
        return {next};
      });
  return loopNest.results.front();
}

static void createSideEffectLoopNest(
    PatternRewriter &rewriter, Location loc, ArrayRef<int64_t> shape,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder) {
  SmallVector<Value> lbs, ubs, steps;
  for (int64_t dim : shape) {
    lbs.push_back(createIndexConstant(rewriter, loc, 0));
    ubs.push_back(createIndexConstant(rewriter, loc, dim));
    steps.push_back(createIndexConstant(rewriter, loc, 1));
  }
  scf::buildLoopNest(rewriter, loc, lbs, ubs, steps,
                     [&](OpBuilder &builder, Location nestedLoc, ValueRange ivs) {
                       bodyBuilder(builder, nestedLoc, ivs);
                     });
}

static SmallVector<Value> computeBaseOffsets(PatternRewriter &rewriter, Location loc,
                                             DenseI64ArrayAttr tile,
                                             ValueRange coords) {
  SmallVector<Value> offsets;
  offsets.reserve(tile.size());
  for (auto [coord, tileDim] : llvm::zip_equal(coords, tile.asArrayRef())) {
    Value coordIndex = asIndex(rewriter, loc, coord);
    Value tileSize = createIndexConstant(rewriter, loc, tileDim);
    offsets.push_back(rewriter.create<arith::MulIOp>(loc, coordIndex, tileSize));
  }
  return offsets;
}

static Value createFloatLikeOne(OpBuilder &builder, Location loc, Type type) {
  return builder.create<arith::ConstantOp>(
      loc, FloatAttr::get(type, 1.0));
}

static Value createFloatLikeZero(OpBuilder &builder, Location loc, Type type) {
  return builder.create<arith::ConstantOp>(
      loc, FloatAttr::get(type, 0.0));
}

static Value createZeroValue(OpBuilder &builder, Location loc, Type type) {
  if (auto floatType = llvm::dyn_cast<FloatType>(type))
    return builder.create<arith::ConstantOp>(loc, FloatAttr::get(floatType, 0.0));
  if (auto intType = llvm::dyn_cast<IntegerType>(type))
    return builder.create<arith::ConstantOp>(loc, IntegerAttr::get(intType, 0));
  llvm_unreachable("unsupported zero element type");
}

static Value castInt8ToType(OpBuilder &builder, Location loc, Value value, Type type) {
  if (type.isF32() || type.isBF16())
    return builder.create<arith::SIToFPOp>(loc, type, value);
  return builder.create<arith::ExtSIOp>(loc, type, value);
}

static SmallVector<Value> computeBroadcastIndices(OpBuilder &builder, Location loc,
                                                  RankedTensorType operandType,
                                                  ValueRange resultIvs) {
  int64_t rankOffset = resultIvs.size() - operandType.getRank();
  SmallVector<Value> indices;
  indices.reserve(operandType.getRank());
  for (int64_t i = 0; i < operandType.getRank(); ++i) {
    if (operandType.getDimSize(i) == 1) {
      indices.push_back(builder.create<arith::ConstantIndexOp>(loc, 0));
      continue;
    }
    indices.push_back(resultIvs[rankOffset + i]);
  }
  return indices;
}

namespace {
struct GetTileBlockIdLowering : public OpRewritePattern<GetTileBlockIdOp> {
  GetTileBlockIdLowering(MLIRContext *context, int32_t x, int32_t y, int32_t z)
      : OpRewritePattern<GetTileBlockIdOp>(context), x(x), y(y), z(z) {}

  LogicalResult matchAndRewrite(GetTileBlockIdOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    rewriter.replaceOp(op,
                       ValueRange{
                           rewriter.create<arith::ConstantIntOp>(loc, x, 32),
                           rewriter.create<arith::ConstantIntOp>(loc, y, 32),
                           rewriter.create<arith::ConstantIntOp>(loc, z, 32),
                       });
    return success();
  }

  int32_t x;
  int32_t y;
  int32_t z;
};

struct LoadLowering : public OpRewritePattern<LoadOp> {
  using OpRewritePattern<LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoadOp op,
                                PatternRewriter &rewriter) const override {
    auto tensorType = llvm::cast<RankedTensorType>(op.getResult().getType());
    Location loc = op.getLoc();
    SmallVector<Value> bases =
        computeBaseOffsets(rewriter, loc, op.getTileAttr(), op.getCoords());

    Value result = createTensorLoopNest(
        rewriter, loc, tensorType,
        [&](OpBuilder &builder, Location nestedLoc, ValueRange ivs) -> Value {
          SmallVector<Value> indices;
          indices.reserve(ivs.size());
          for (auto [base, iv] : llvm::zip_equal(bases, ivs))
            indices.push_back(builder.create<arith::AddIOp>(nestedLoc, base, iv));
          return builder.create<memref::LoadOp>(nestedLoc, op.getSource(), indices);
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct StoreLowering : public OpRewritePattern<StoreOp> {
  using OpRewritePattern<StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StoreOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto tensorType = llvm::cast<RankedTensorType>(op.getValue().getType());
    SmallVector<Value> bases =
        computeBaseOffsets(rewriter, loc, op.getTileAttr(), op.getCoords());

    createSideEffectLoopNest(
        rewriter, loc, tensorType.getShape(),
        [&](OpBuilder &builder, Location nestedLoc, ValueRange ivs) {
          Value src =
              builder.create<tensor::ExtractOp>(nestedLoc, op.getValue(), ivs);
          SmallVector<Value> indices;
          indices.reserve(ivs.size());
          for (auto [base, iv] : llvm::zip_equal(bases, ivs))
            indices.push_back(builder.create<arith::AddIOp>(nestedLoc, base, iv));
          builder.create<memref::StoreOp>(nestedLoc, src, op.getDest(), indices);
        });
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename OpTy>
struct UnaryElementwiseLowering : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    auto tensorType = llvm::cast<RankedTensorType>(op.getResult().getType());
    Location loc = op.getLoc();
    Value result = createTensorLoopNest(
        rewriter, loc, tensorType,
        [&](OpBuilder &builder, Location nestedLoc, ValueRange ivs) -> Value {
          Value input =
              builder.create<tensor::ExtractOp>(nestedLoc, op.getInput(), ivs);
          if constexpr (std::is_same_v<OpTy, ExpOp>)
            return builder.create<math::ExpOp>(nestedLoc, input);
          if constexpr (std::is_same_v<OpTy, CosOp>)
            return builder.create<math::CosOp>(nestedLoc, input);
          if constexpr (std::is_same_v<OpTy, SinOp>)
            return builder.create<math::SinOp>(nestedLoc, input);
          if constexpr (std::is_same_v<OpTy, TanhOp>)
            return builder.create<math::TanhOp>(nestedLoc, input);
          if constexpr (std::is_same_v<OpTy, ReciprocalOp>) {
            Value one = createFloatLikeOne(builder, nestedLoc, tensorType.getElementType());
            return builder.create<arith::DivFOp>(nestedLoc, one, input);
          }
          if constexpr (std::is_same_v<OpTy, RsqrtOp>) {
            Value one = createFloatLikeOne(builder, nestedLoc, tensorType.getElementType());
            Value sqrt = builder.create<math::SqrtOp>(nestedLoc, input);
            return builder.create<arith::DivFOp>(nestedLoc, one, sqrt);
          }
          if constexpr (std::is_same_v<OpTy, SigmoidOp>) {
            Value zero = createFloatLikeZero(builder, nestedLoc, tensorType.getElementType());
            Value one = createFloatLikeOne(builder, nestedLoc, tensorType.getElementType());
            Value neg = builder.create<arith::SubFOp>(nestedLoc, zero, input);
            Value exp = builder.create<math::ExpOp>(nestedLoc, neg);
            Value denom = builder.create<arith::AddFOp>(nestedLoc, one, exp);
            return builder.create<arith::DivFOp>(nestedLoc, one, denom);
          }
          if constexpr (std::is_same_v<OpTy, SiluOp>) {
            Value zero = createFloatLikeZero(builder, nestedLoc, tensorType.getElementType());
            Value one = createFloatLikeOne(builder, nestedLoc, tensorType.getElementType());
            Value neg = builder.create<arith::SubFOp>(nestedLoc, zero, input);
            Value exp = builder.create<math::ExpOp>(nestedLoc, neg);
            Value denom = builder.create<arith::AddFOp>(nestedLoc, one, exp);
            Value sigmoid = builder.create<arith::DivFOp>(nestedLoc, one, denom);
            return builder.create<arith::MulFOp>(nestedLoc, input, sigmoid);
          }
          llvm_unreachable("unsupported unary op");
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename OpTy>
struct BinaryElementwiseLowering : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    auto tensorType = llvm::cast<RankedTensorType>(op.getResult().getType());
    auto lhsType = llvm::cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType = llvm::cast<RankedTensorType>(op.getRhs().getType());
    Location loc = op.getLoc();
    Value result = createTensorLoopNest(
        rewriter, loc, tensorType,
        [&](OpBuilder &builder, Location nestedLoc, ValueRange ivs) -> Value {
          SmallVector<Value> lhsIvs =
              computeBroadcastIndices(builder, nestedLoc, lhsType, ivs);
          SmallVector<Value> rhsIvs =
              computeBroadcastIndices(builder, nestedLoc, rhsType, ivs);
          Value lhs = builder.create<tensor::ExtractOp>(nestedLoc, op.getLhs(), lhsIvs);
          Value rhs = builder.create<tensor::ExtractOp>(nestedLoc, op.getRhs(), rhsIvs);
          bool isFloat = llvm::isa<FloatType>(tensorType.getElementType());
          if constexpr (std::is_same_v<OpTy, AddOp>) {
            if (isFloat)
              return builder.create<arith::AddFOp>(nestedLoc, lhs, rhs).getResult();
            return builder.create<arith::AddIOp>(nestedLoc, lhs, rhs).getResult();
          }
          if constexpr (std::is_same_v<OpTy, SubOp>) {
            if (isFloat)
              return builder.create<arith::SubFOp>(nestedLoc, lhs, rhs).getResult();
            return builder.create<arith::SubIOp>(nestedLoc, lhs, rhs).getResult();
          }
          if constexpr (std::is_same_v<OpTy, MulOp>) {
            if (isFloat)
              return builder.create<arith::MulFOp>(nestedLoc, lhs, rhs).getResult();
            return builder.create<arith::MulIOp>(nestedLoc, lhs, rhs).getResult();
          }
          llvm_unreachable("unsupported binary op");
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct MatmulLowering : public OpRewritePattern<MatmulOp> {
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = llvm::cast<RankedTensorType>(op.getResult().getType());
    auto lhsType = llvm::cast<RankedTensorType>(op.getLhs().getType());
    auto lhsElemType = lhsType.getElementType();
    auto rhsElemType =
        llvm::cast<RankedTensorType>(op.getRhs().getType()).getElementType();
    auto resultElemType = resultType.getElementType();
    bool resultIsFloat = llvm::isa<FloatType>(resultElemType);
    Location loc = op.getLoc();
    Value result = createTensorLoopNest(
        rewriter, loc, resultType,
        [&](OpBuilder &builder, Location nestedLoc, ValueRange ivs) -> Value {
          Value acc = createZeroValue(builder, nestedLoc, resultElemType);
          for (int64_t k = 0; k < lhsType.getDimSize(1); ++k) {
            Value kVal = createIndexConstant(rewriter, nestedLoc, k);
            Value lhs = builder.create<tensor::ExtractOp>(
                nestedLoc, op.getLhs(), ValueRange{ivs[0], kVal});
            Value rhs = builder.create<tensor::ExtractOp>(
                nestedLoc, op.getRhs(), ValueRange{kVal, ivs[1]});
            if (lhsElemType.isInteger(8) && rhsElemType.isInteger(8) && resultIsFloat) {
              lhs = castInt8ToType(builder, nestedLoc, lhs, resultElemType);
              rhs = castInt8ToType(builder, nestedLoc, rhs, resultElemType);
            }
            Value prod;
            if (resultIsFloat)
              prod = builder.create<arith::MulFOp>(nestedLoc, lhs, rhs).getResult();
            else
              prod = builder.create<arith::MulIOp>(nestedLoc, lhs, rhs).getResult();
            if (resultIsFloat)
              acc = builder.create<arith::AddFOp>(nestedLoc, acc, prod).getResult();
            else
              acc = builder.create<arith::AddIOp>(nestedLoc, acc, prod).getResult();
          }
          return acc;
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct MMALowering : public OpRewritePattern<MMAOp> {
  using OpRewritePattern<MMAOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MMAOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = llvm::cast<RankedTensorType>(op.getResult().getType());
    auto lhsType = llvm::cast<RankedTensorType>(op.getLhs().getType());
    Location loc = op.getLoc();
    SmallVector<Value> lbs = {createIndexConstant(rewriter, loc, 0),
                              createIndexConstant(rewriter, loc, 0)};
    SmallVector<Value> ubs = {createIndexConstant(rewriter, loc, resultType.getDimSize(0)),
                              createIndexConstant(rewriter, loc, resultType.getDimSize(1))};
    SmallVector<Value> steps = {createIndexConstant(rewriter, loc, 1),
                                createIndexConstant(rewriter, loc, 1)};

    Value result = scf::buildLoopNest(
                       rewriter, loc, lbs, ubs, steps, ValueRange{op.getAcc()},
                       [&](OpBuilder &builder, Location nestedLoc, ValueRange ivs,
                           ValueRange iterArgs) -> scf::ValueVector {
                         Value acc = builder.create<tensor::ExtractOp>(nestedLoc, iterArgs[0], ivs);
                         for (int64_t k = 0; k < lhsType.getDimSize(1); ++k) {
                           Value kVal = createIndexConstant(rewriter, nestedLoc, k);
                           Value lhs = builder.create<tensor::ExtractOp>(
                               nestedLoc, op.getLhs(), ValueRange{ivs[0], kVal});
                           Value rhs = builder.create<tensor::ExtractOp>(
                               nestedLoc, op.getRhs(), ValueRange{kVal, ivs[1]});
                           Value lhsCast = castInt8ToType(builder, nestedLoc, lhs,
                                                          resultType.getElementType());
                           Value rhsCast = castInt8ToType(builder, nestedLoc, rhs,
                                                          resultType.getElementType());
                           Value prod = builder.create<arith::MulFOp>(nestedLoc, lhsCast, rhsCast);
                           acc = builder.create<arith::AddFOp>(nestedLoc, acc, prod);
                         }
                         Value next = builder.create<tensor::InsertOp>(nestedLoc, acc, iterArgs[0], ivs);
                         return {next};
                       })
                       .results.front();
    rewriter.replaceOp(op, result);
    return success();
  }
};

class XTLowerToLoopsPass
    : public mlir::xt::impl::XTLowerToLoopsBase<XTLowerToLoopsPass> {
public:
  using mlir::xt::impl::XTLowerToLoopsBase<XTLowerToLoopsPass>::XTLowerToLoopsBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect, math::MathDialect, memref::MemRefDialect,
                           scf::SCFDialect, tensor::TensorDialect>();
    target.addIllegalDialect<XTDialect>();

    RewritePatternSet patterns(context);
    patterns.add<GetTileBlockIdLowering>(context, blockIdX, blockIdY, blockIdZ);
    patterns.add<LoadLowering, StoreLowering, BinaryElementwiseLowering<AddOp>,
                 BinaryElementwiseLowering<SubOp>, BinaryElementwiseLowering<MulOp>,
                 UnaryElementwiseLowering<ExpOp>,
                 UnaryElementwiseLowering<CosOp>, UnaryElementwiseLowering<SinOp>,
                 UnaryElementwiseLowering<ReciprocalOp>,
                 UnaryElementwiseLowering<RsqrtOp>,
                 UnaryElementwiseLowering<SigmoidOp>,
                 UnaryElementwiseLowering<TanhOp>,
                 UnaryElementwiseLowering<SiluOp>, MatmulLowering,
                 MMALowering>(context);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::xt::createXTLowerToLoopsPass() {
  return std::make_unique<XTLowerToLoopsPass>();
}
