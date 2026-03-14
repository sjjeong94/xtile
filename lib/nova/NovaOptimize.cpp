#include "nova/NovaPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::nova {
#define GEN_PASS_DEF_NOVAOPTIMIZE
#include "nova/NovaPasses.h.inc"
} // namespace mlir::nova

using namespace mlir;

namespace {
static FailureOr<float> getSplatFloatValue(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return failure();

  Attribute attr = constant.getValue();
  auto dense = llvm::dyn_cast<DenseElementsAttr>(attr);
  if (!dense || !dense.isSplat())
    return failure();

  auto splat = dense.getSplatValue<Attribute>();
  auto floatAttr = llvm::dyn_cast<FloatAttr>(splat);
  if (!floatAttr)
    return failure();
  return floatAttr.getValueAsDouble();
}

static float getFloatAttr(Operation *op, StringRef name) {
  return op->getAttrOfType<FloatAttr>(name).getValueAsDouble();
}

static FloatAttr makeFloatAttr(MLIRContext *context, float value) {
  return FloatAttr::get(Float32Type::get(context), value);
}

static Value buildScalarTensorConstant(PatternRewriter &rewriter, Location loc,
                                       float value) {
  auto tensorType = RankedTensorType::get({1, 1}, rewriter.getF32Type());
  return rewriter
      .create<arith::ConstantOp>(loc, DenseElementsAttr::get(tensorType, value))
      .getResult();
}

static bool isFloatAttrValue(Operation *op, StringRef name, float value) {
  return getFloatAttr(op, name) == value;
}

static LogicalResult foldScalarIntoSide(Operation *scalarOp, float &scale,
                                        float &bias) {
  int32_t mode =
      scalarOp->getAttrOfType<IntegerAttr>("mode").getInt();
  float rhs = getFloatAttr(scalarOp, "rhs");
  if (mode == 3) {
    mode = 1;
    rhs = -rhs;
  }
  switch (mode) {
  case 1:
    bias += rhs * scale;
    return success();
  case 2:
    scale *= rhs;
    return success();
  default:
    return failure();
  }
}

template <typename OpTy>
struct FoldScalarIntoBinaryPattern : OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    float lhsScale = getFloatAttr(op, "lhs_s");
    float lhsBias = getFloatAttr(op, "lhs_b");
    float rhsScale = getFloatAttr(op, "rhs_s");
    float rhsBias = getFloatAttr(op, "rhs_b");

    bool changed = false;
    if (Operation *lhsDef = lhs.getDefiningOp();
        lhsDef && lhsDef->getName().getStringRef() == "nova.scalar") {
      if (succeeded(foldScalarIntoSide(lhsDef, lhsScale, lhsBias))) {
        lhs = lhsDef->getOperand(0);
        changed = true;
      }
    }
    if (Operation *rhsDef = rhs.getDefiningOp();
        rhsDef && rhsDef->getName().getStringRef() == "nova.scalar") {
      if (succeeded(foldScalarIntoSide(rhsDef, rhsScale, rhsBias))) {
        rhs = rhsDef->getOperand(0);
        changed = true;
      }
    }
    if (!changed)
      return failure();

    OperationState state(op.getLoc(), op->getName().getStringRef());
    state.addOperands({lhs, rhs});
    state.addTypes(op.getResult().getType());
    state.addAttribute("lhs_b", makeFloatAttr(rewriter.getContext(), lhsBias));
    state.addAttribute("lhs_s", makeFloatAttr(rewriter.getContext(), lhsScale));
    state.addAttribute("mode", op->getAttr("mode"));
    state.addAttribute("rhs_b", makeFloatAttr(rewriter.getContext(), rhsBias));
    state.addAttribute("rhs_s", makeFloatAttr(rewriter.getContext(), rhsScale));

    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct FoldScalarIntoMatmulPattern : OpRewritePattern<nova::ScalarOp> {
  using OpRewritePattern<nova::ScalarOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(nova::ScalarOp op,
                                PatternRewriter &rewriter) const override {
    auto matmulOp = op.getInput().getDefiningOp<nova::MatmulOp>();
    if (!matmulOp)
      return failure();

    FailureOr<float> scaleValue = getSplatFloatValue(matmulOp.getScale());
    FailureOr<float> biasValue = getSplatFloatValue(matmulOp.getBias());
    if (failed(scaleValue) || failed(biasValue))
      return failure();

    int32_t mode = op->getAttrOfType<IntegerAttr>("mode").getInt();
    float rhs = getFloatAttr(op, "rhs");

    Value newScale = matmulOp.getScale();
    Value newBias = matmulOp.getBias();
    switch (mode) {
    case 1:
      newBias = buildScalarTensorConstant(rewriter, op.getLoc(),
                                          *biasValue + rhs);
      break;
    case 2:
      newScale = buildScalarTensorConstant(rewriter, op.getLoc(),
                                           *scaleValue * rhs);
      newBias = buildScalarTensorConstant(rewriter, op.getLoc(),
                                          *biasValue * rhs);
      break;
    default:
      return failure();
    }

    OperationState state(op.getLoc(), "nova.matmul");
    state.addOperands(
        {matmulOp.getLhs(), matmulOp.getRhs(), newScale, newBias});
    state.addTypes(op.getResult().getType());

    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct FoldBroadcastMulIntoMatmulPattern
    : OpRewritePattern<nova::BroadcastOp> {
  using OpRewritePattern<nova::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(nova::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    int32_t mode = op->getAttrOfType<IntegerAttr>("mode").getInt();
    if (mode != 2)
      return failure();

    auto matmulOp = op.getLhs().getDefiningOp<nova::MatmulOp>();
    if (!matmulOp || !matmulOp->hasOneUse())
      return failure();

    FailureOr<float> scaleValue = getSplatFloatValue(matmulOp.getScale());
    FailureOr<float> biasValue = getSplatFloatValue(matmulOp.getBias());
    if (failed(scaleValue) || failed(biasValue) || *scaleValue != 1.0f ||
        *biasValue != 0.0f)
      return failure();

    if (!isFloatAttrValue(op, "lhs_s", 1.0f) || !isFloatAttrValue(op, "lhs_b", 0.0f) ||
        !isFloatAttrValue(op, "rhs_s", 1.0f) || !isFloatAttrValue(op, "rhs_b", 0.0f))
      return failure();

    OperationState state(op.getLoc(), "nova.matmul");
    state.addOperands(
        {matmulOp.getLhs(), matmulOp.getRhs(), op.getRhs(), matmulOp.getBias()});
    state.addTypes(op.getResult().getType());

    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct FoldBroadcastAddIntoMatmulPattern
    : OpRewritePattern<nova::BroadcastOp> {
  using OpRewritePattern<nova::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(nova::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    int32_t mode = op->getAttrOfType<IntegerAttr>("mode").getInt();
    if (mode != 1)
      return failure();

    auto matmulOp = op.getLhs().getDefiningOp<nova::MatmulOp>();
    if (!matmulOp || !matmulOp->hasOneUse())
      return failure();

    FailureOr<float> biasValue = getSplatFloatValue(matmulOp.getBias());
    if (failed(biasValue) || *biasValue != 0.0f)
      return failure();

    if (!isFloatAttrValue(op, "lhs_s", 1.0f) ||
        !isFloatAttrValue(op, "lhs_b", 0.0f) ||
        !isFloatAttrValue(op, "rhs_s", 1.0f) ||
        !isFloatAttrValue(op, "rhs_b", 0.0f))
      return failure();

    OperationState state(op.getLoc(), "nova.matmul");
    state.addOperands(
        {matmulOp.getLhs(), matmulOp.getRhs(), matmulOp.getScale(), op.getRhs()});
    state.addTypes(op.getResult().getType());

    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

class NovaOptimizePass
    : public mlir::nova::impl::NovaOptimizeBase<NovaOptimizePass> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldScalarIntoBinaryPattern<nova::BroadcastOp>,
                 FoldScalarIntoBinaryPattern<nova::ElementwiseOp>,
                 FoldScalarIntoMatmulPattern,
                 FoldBroadcastMulIntoMatmulPattern,
                 FoldBroadcastAddIntoMatmulPattern>(
        &getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::nova::createNovaOptimizePass() {
  return std::make_unique<NovaOptimizePass>();
}
