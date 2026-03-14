#include "nova/NovaPasses.h"

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
static float getFloatAttr(Operation *op, StringRef name) {
  return op->getAttrOfType<FloatAttr>(name).getValueAsDouble();
}

static FloatAttr makeFloatAttr(MLIRContext *context, float value) {
  return FloatAttr::get(Float32Type::get(context), value);
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

class NovaOptimizePass
    : public mlir::nova::impl::NovaOptimizeBase<NovaOptimizePass> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldScalarIntoBinaryPattern<nova::BroadcastOp>,
                 FoldScalarIntoBinaryPattern<nova::ElementwiseOp>>(
        &getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::nova::createNovaOptimizePass() {
  return std::make_unique<NovaOptimizePass>();
}
