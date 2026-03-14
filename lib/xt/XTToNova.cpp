#include "xt/XTPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "nova/NovaDialect.h"
#include "xt/XTOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::xt {
#define GEN_PASS_DEF_XTTONOVA
#include "xt/XTPasses.h.inc"
} // namespace mlir::xt

using namespace mlir;

namespace {
static IntegerAttr buildModeAttr(MLIRContext *context, int32_t mode) {
  return IntegerAttr::get(IntegerType::get(context, 32), mode);
}

static FloatAttr buildFloatAttr(MLIRContext *context, float value) {
  return FloatAttr::get(Float32Type::get(context), value);
}

template <typename OpTy>
struct BinaryOpToNovaPattern : OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    auto lhsType = dyn_cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType = dyn_cast<RankedTensorType>(op.getRhs().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!lhsType || !rhsType || !resultType)
      return failure();

    bool isElementwise = lhsType == resultType && rhsType == resultType;
    StringRef novaName =
        isElementwise ? "nova.elementwise" : "nova.broadcast";

    OperationState state(op.getLoc(), novaName);
    state.addOperands({op.getLhs(), op.getRhs()});
    state.addTypes(resultType);
    state.addAttribute("mode", buildModeAttr(rewriter.getContext(), getMode()));
    state.addAttribute("lhs_s", buildFloatAttr(rewriter.getContext(), 1.0f));
    state.addAttribute("lhs_b", buildFloatAttr(rewriter.getContext(), 0.0f));
    state.addAttribute("rhs_s", buildFloatAttr(rewriter.getContext(), 1.0f));
    state.addAttribute("rhs_b", buildFloatAttr(rewriter.getContext(), 0.0f));

    Operation *novaOp = rewriter.create(state);
    rewriter.replaceOp(op, novaOp->getResults());
    return success();
  }

  static int32_t getMode();
};

template <typename OpTy>
struct ReduceOpToNovaPattern : OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!inputType || !resultType)
      return failure();

    OperationState state(op.getLoc(), "nova.reduce");
    state.addOperands(op.getInput());
    state.addTypes(resultType);
    state.addAttribute("mode", buildModeAttr(rewriter.getContext(), getMode()));

    Operation *novaOp = rewriter.create(state);
    rewriter.replaceOp(op, novaOp->getResults());
    return success();
  }

  static int32_t getMode();
};

template <>
int32_t BinaryOpToNovaPattern<xt::AddOp>::getMode() {
  return 1;
}

template <>
int32_t BinaryOpToNovaPattern<xt::MulOp>::getMode() {
  return 2;
}

template <>
int32_t BinaryOpToNovaPattern<xt::SubOp>::getMode() {
  return 3;
}

template <>
int32_t ReduceOpToNovaPattern<xt::ReduceSumOp>::getMode() {
  return 0;
}

template <>
int32_t ReduceOpToNovaPattern<xt::ReduceMaxOp>::getMode() {
  return 1;
}

class XTToNovaPass : public mlir::xt::impl::XTToNovaBase<XTToNovaPass> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<BinaryOpToNovaPattern<xt::AddOp>,
                 BinaryOpToNovaPattern<xt::MulOp>,
                 BinaryOpToNovaPattern<xt::SubOp>,
                 ReduceOpToNovaPattern<xt::ReduceSumOp>,
                 ReduceOpToNovaPattern<xt::ReduceMaxOp>>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::xt::createXTToNovaPass() {
  return std::make_unique<XTToNovaPass>();
}
