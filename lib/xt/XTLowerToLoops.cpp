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

static Value createTileLoopNest(PatternRewriter &rewriter, Location loc,
                                RankedTensorType tensorType,
                                function_ref<Value(OpBuilder &, Location, Value, Value)> bodyBuilder) {
  auto shape = tensorType.getShape();
  Value empty = tensor::EmptyOp::create(rewriter, loc, shape,
                                        tensorType.getElementType());
  SmallVector<Value> lbs = {createIndexConstant(rewriter, loc, 0),
                            createIndexConstant(rewriter, loc, 0)};
  SmallVector<Value> ubs = {createIndexConstant(rewriter, loc, shape[0]),
                            createIndexConstant(rewriter, loc, shape[1])};
  SmallVector<Value> steps = {createIndexConstant(rewriter, loc, 1),
                              createIndexConstant(rewriter, loc, 1)};
  scf::LoopNest loopNest = scf::buildLoopNest(
      rewriter, loc, lbs, ubs, steps, ValueRange{empty},
      [&](OpBuilder &builder, Location nestedLoc, ValueRange ivs,
          ValueRange iterArgs) -> scf::ValueVector {
        Value elem = bodyBuilder(builder, nestedLoc, ivs[0], ivs[1]);
        Value next = builder.create<tensor::InsertOp>(nestedLoc, elem, iterArgs[0], ivs);
        return {next};
      });
  return loopNest.results.front();
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
    auto tile = op.getTile();
    Value tileX = asIndex(rewriter, loc, op.getTileX());
    Value tileY = asIndex(rewriter, loc, op.getTileY());
    Value tileH = createIndexConstant(rewriter, loc, tile[0]);
    Value tileW = createIndexConstant(rewriter, loc, tile[1]);
    Value rowBase = rewriter.create<arith::MulIOp>(loc, tileX, tileH);
    Value colBase = rewriter.create<arith::MulIOp>(loc, tileY, tileW);

    Value result = createTileLoopNest(
        rewriter, loc, tensorType,
        [&](OpBuilder &builder, Location nestedLoc, Value row, Value col) -> Value {
          Value srcRow = builder.create<arith::AddIOp>(nestedLoc, rowBase, row);
          Value srcCol = builder.create<arith::AddIOp>(nestedLoc, colBase, col);
          return builder.create<memref::LoadOp>(nestedLoc, op.getSource(),
                                                ValueRange{srcRow, srcCol});
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
    auto tile = op.getTile();
    Value tileX = asIndex(rewriter, loc, op.getTileX());
    Value tileY = asIndex(rewriter, loc, op.getTileY());
    Value tileH = createIndexConstant(rewriter, loc, tile[0]);
    Value tileW = createIndexConstant(rewriter, loc, tile[1]);
    Value rowBase = rewriter.create<arith::MulIOp>(loc, tileX, tileH);
    Value colBase = rewriter.create<arith::MulIOp>(loc, tileY, tileW);

    SmallVector<Value> lbs = {createIndexConstant(rewriter, loc, 0),
                              createIndexConstant(rewriter, loc, 0)};
    SmallVector<Value> ubs = {createIndexConstant(rewriter, loc, tile[0]),
                              createIndexConstant(rewriter, loc, tile[1])};
    SmallVector<Value> steps = {createIndexConstant(rewriter, loc, 1),
                                createIndexConstant(rewriter, loc, 1)};
    scf::buildLoopNest(rewriter, loc, lbs, ubs, steps,
                       [&](OpBuilder &builder, Location nestedLoc, ValueRange ivs) {
                         Value src = builder.create<tensor::ExtractOp>(
                             nestedLoc, op.getValue(), ivs);
                         Value dstRow = builder.create<arith::AddIOp>(nestedLoc, rowBase, ivs[0]);
                         Value dstCol = builder.create<arith::AddIOp>(nestedLoc, colBase, ivs[1]);
                         builder.create<memref::StoreOp>(nestedLoc, src, op.getDest(),
                                                         ValueRange{dstRow, dstCol});
                       });
    rewriter.eraseOp(op);
    return success();
  }
};

struct AddLowering : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    auto tensorType = llvm::cast<RankedTensorType>(op.getResult().getType());
    Location loc = op.getLoc();
    Value result = createTileLoopNest(
        rewriter, loc, tensorType,
        [&](OpBuilder &builder, Location nestedLoc, Value row, Value col) -> Value {
          Value lhs = builder.create<tensor::ExtractOp>(nestedLoc, op.getLhs(),
                                                        ValueRange{row, col});
          Value rhs = builder.create<tensor::ExtractOp>(nestedLoc, op.getRhs(),
                                                        ValueRange{row, col});
          if (llvm::isa<FloatType>(tensorType.getElementType()))
            return builder.create<arith::AddFOp>(nestedLoc, lhs, rhs);
          return builder.create<arith::AddIOp>(nestedLoc, lhs, rhs);
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ExpLowering : public OpRewritePattern<ExpOp> {
  using OpRewritePattern<ExpOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExpOp op,
                                PatternRewriter &rewriter) const override {
    auto tensorType = llvm::cast<RankedTensorType>(op.getResult().getType());
    if (!llvm::isa<FloatType>(tensorType.getElementType()))
      return rewriter.notifyMatchFailure(op, "exp lowering requires floating element type");
    Location loc = op.getLoc();
    Value result = createTileLoopNest(
        rewriter, loc, tensorType,
        [&](OpBuilder &builder, Location nestedLoc, Value row, Value col) -> Value {
          Value input = builder.create<tensor::ExtractOp>(nestedLoc, op.getInput(),
                                                          ValueRange{row, col});
          return builder.create<math::ExpOp>(nestedLoc, input);
        });
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
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect, math::MathDialect,
                           memref::MemRefDialect, scf::SCFDialect,
                           tensor::TensorDialect>();
    target.addIllegalDialect<XTDialect>();

    RewritePatternSet patterns(context);
    patterns.add<GetTileBlockIdLowering>(context, blockIdX, blockIdY, blockIdZ);
    patterns.add<LoadLowering, StoreLowering, AddLowering, ExpLowering>(context);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::xt::createXTLowerToLoopsPass() {
  return std::make_unique<XTLowerToLoopsPass>();
}
