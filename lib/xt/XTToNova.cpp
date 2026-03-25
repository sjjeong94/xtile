#include "xt/XTPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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

static IntegerAttr buildAxisAttr(MLIRContext *context, int64_t axis) {
  return IntegerAttr::get(IntegerType::get(context, 64), axis);
}

static FloatAttr buildFloatAttr(MLIRContext *context, float value) {
  return FloatAttr::get(Float32Type::get(context), value);
}

static std::pair<int32_t, float> normalizeScalarModeAndRhs(int32_t mode,
                                                           float rhs) {
  if (mode == 3)
    return {1, -rhs};
  return {mode, rhs};
}

static bool hasSingleElement(RankedTensorType type) {
  if (!type || !type.hasStaticShape())
    return false;

  int64_t numElements = 1;
  for (int64_t dim : type.getShape())
    numElements *= dim;
  return numElements == 1;
}

static FailureOr<float> extractConstantFloat(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return failure();

  Attribute attr = constant.getValue();
  if (auto floatAttr = llvm::dyn_cast<FloatAttr>(attr))
    return floatAttr.getValueAsDouble();

  auto dense = llvm::dyn_cast<DenseElementsAttr>(attr);
  if (!dense || !dense.isSplat())
    return failure();

  auto splat = dense.getSplatValue<Attribute>();
  auto floatAttr = llvm::dyn_cast<FloatAttr>(splat);
  if (!floatAttr)
    return failure();
  return floatAttr.getValueAsDouble();
}

static FailureOr<int64_t> extractConstantInt(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return failure();

  Attribute attr = constant.getValue();
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getInt();

  return failure();
}

static FailureOr<DenseI64ArrayAttr> extractConstantStarts(MLIRContext *context,
                                                          ValueRange values,
                                                          RankedTensorType type) {
  if (!type || !type.hasStaticShape())
    return failure();

  ArrayRef<int64_t> shape = type.getShape();
  if (shape.size() != values.size())
    return failure();

  SmallVector<int64_t> starts;
  starts.reserve(values.size());
  for (auto [value, dim] : llvm::zip(values, shape)) {
    FailureOr<int64_t> index = extractConstantInt(value);
    if (failed(index))
      return failure();
    starts.push_back(*index * dim);
  }
  return DenseI64ArrayAttr::get(context, starts);
}

struct Conv2DSpatialSlice {
  int64_t loadStart;
  int64_t loadSize;
  int64_t convPadBefore;
  int64_t convPadAfter;
};

static FailureOr<Conv2DSpatialSlice>
computeConv2DSpatialSlice(int64_t sourceDim, int64_t resultIndex,
                          int64_t resultDim, int64_t kernelDim,
                          int64_t padBefore, int64_t padAfter, int64_t stride,
                          int64_t dilation) {
  if (sourceDim < 0 || resultIndex < 0 || resultDim < 0 || kernelDim <= 0 ||
      padBefore < 0 || padAfter < 0 || stride <= 0 || dilation <= 0)
    return failure();

  int64_t resultStart = resultIndex * resultDim;
  int64_t receptiveField =
      (resultDim - 1) * stride + dilation * (kernelDim - 1) + 1;
  int64_t idealStart = resultStart * stride - padBefore;
  int64_t idealEnd = idealStart + receptiveField;
  int64_t loadStart = std::max<int64_t>(idealStart, 0);
  int64_t convPadBefore = loadStart - idealStart;
  int64_t unclampedEnd = loadStart + (receptiveField - convPadBefore);
  int64_t loadEnd = std::min<int64_t>(unclampedEnd, sourceDim);
  int64_t loadSize = loadEnd - loadStart;
  int64_t convPadAfter = idealEnd - loadEnd;
  (void)padAfter;
  if (loadSize <= 0)
    return failure();
  if (convPadAfter < 0)
    return failure();

  return Conv2DSpatialSlice{
      loadStart,
      loadSize,
      convPadBefore,
      convPadAfter,
  };
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
    if constexpr (std::is_same_v<OpTy, xt::MulOp>) {
      if (op.getLhs() == op.getRhs()) {
        OperationState state(op.getLoc(), "nova.square");
        state.addOperands(op.getLhs());
        state.addTypes(resultType);

        Operation *novaOp = rewriter.create(state);
        rewriter.replaceOp(op, novaOp->getResults());
        return success();
      }
    }
    FailureOr<float> rhsConstant = extractConstantFloat(op.getRhs());
    if (succeeded(rhsConstant)) {
      auto [mode, rhsValue] = normalizeScalarModeAndRhs(getMode(), *rhsConstant);
      OperationState state(op.getLoc(), "nova.scalar");
      state.addOperands(op.getLhs());
      state.addTypes(resultType);
      state.addAttribute("mode", buildModeAttr(rewriter.getContext(), mode));
      state.addAttribute("rhs",
                         buildFloatAttr(rewriter.getContext(), rhsValue));

      Operation *novaOp = rewriter.create(state);
      rewriter.replaceOp(op, novaOp->getResults());
      return success();
    }
    if (hasSingleElement(lhsType) || hasSingleElement(rhsType))
      return failure();

    bool isElementwise = lhsType == resultType && rhsType == resultType;
    StringRef novaName =
        isElementwise ? "nova.elementwise" : "nova.broadcast";

    OperationState state(op.getLoc(), novaName);
    state.addOperands({op.getLhs(), op.getRhs()});
    state.addTypes(resultType);
    state.addAttribute("mode", buildModeAttr(rewriter.getContext(), getMode()));
    state.addAttribute("lhs_a", buildFloatAttr(rewriter.getContext(), 1.0f));
    state.addAttribute("lhs_b", buildFloatAttr(rewriter.getContext(), 0.0f));
    state.addAttribute("rhs_a", buildFloatAttr(rewriter.getContext(), 1.0f));
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
    state.addAttribute("axis", buildAxisAttr(rewriter.getContext(), getAxis(op)));
    state.addAttribute("mode", buildModeAttr(rewriter.getContext(), getMode()));

    Operation *novaOp = rewriter.create(state);
    rewriter.replaceOp(op, novaOp->getResults());
    return success();
  }

  static int32_t getMode();
  static int64_t getAxis(OpTy op) { return op.getAxis(); }
};

template <typename OpTy>
struct UnaryTensorOpToNovaPattern : OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!inputType || !resultType)
      return failure();

    OperationState state(op.getLoc(), getNovaName());
    state.addOperands(op.getInput());
    state.addTypes(resultType);

    Operation *novaOp = rewriter.create(state);
    rewriter.replaceOp(op, novaOp->getResults());
    return success();
  }

  static StringRef getNovaName();
};

template <typename OpTy>
struct UnaryCastOpToNovaPattern : OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!inputType || !resultType)
      return failure();

    OperationState state(op.getLoc(), getNovaName());
    state.addOperands(op.getInput());
    state.addTypes(resultType);

    Operation *novaOp = rewriter.create(state);
    rewriter.replaceOp(op, novaOp->getResults());
    return success();
  }

  static StringRef getNovaName();
};

struct PermuteOpToNovaPattern : OpRewritePattern<xt::PermuteOp> {
  using OpRewritePattern<xt::PermuteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xt::PermuteOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!inputType || !resultType)
      return failure();

    OperationState state(op.getLoc(), "nova.permute");
    state.addOperands(op.getInput());
    state.addTypes(resultType);
    state.addAttribute("permutation", op.getPermutationAttr());

    Operation *novaOp = rewriter.create(state);
    rewriter.replaceOp(op, novaOp->getResults());
    return success();
  }
};

struct MatmulOpToNovaPattern : OpRewritePattern<xt::MatmulOp> {
  using OpRewritePattern<xt::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xt::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsType = dyn_cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType = dyn_cast<RankedTensorType>(op.getRhs().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!lhsType || !rhsType || !resultType)
      return failure();

    auto scalarTensorType =
        RankedTensorType::get({1, 1}, rewriter.getF32Type());
    auto scale = rewriter.create<arith::ConstantOp>(
        op.getLoc(), DenseElementsAttr::get(scalarTensorType, 1.0f));
    auto bias = rewriter.create<arith::ConstantOp>(
        op.getLoc(), DenseElementsAttr::get(scalarTensorType, 0.0f));

    OperationState state(op.getLoc(), "nova.matmul");
    state.addOperands({op.getLhs(), op.getRhs(), scale.getResult(),
                       bias.getResult()});
    state.addTypes(resultType);

    Operation *novaOp = rewriter.create(state);
    rewriter.replaceOp(op, novaOp->getResults());
    return success();
  }
};

struct LoadOpToNovaPattern : OpRewritePattern<xt::LoadOp> {
  using OpRewritePattern<xt::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xt::LoadOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultType)
      return failure();

    FailureOr<DenseI64ArrayAttr> startAttr = extractConstantStarts(
        rewriter.getContext(), op.getCoords(), resultType);
    if (failed(startAttr))
      return failure();

    OperationState state(op.getLoc(), "nova.load");
    state.addOperands(op.getSource());
    state.addTypes(resultType);
    state.addAttribute("start", *startAttr);
    if (auto shared = op.getSharedAttr())
      state.addAttribute("shared", shared);

    Operation *novaOp = rewriter.create(state);
    rewriter.replaceOp(op, novaOp->getResults());
    return success();
  }
};

struct LoadConv2DOpToNovaPattern : OpRewritePattern<xt::LoadConv2DOp> {
  using OpRewritePattern<xt::LoadConv2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xt::LoadConv2DOp op,
                                PatternRewriter &rewriter) const override {
    auto memRefType = dyn_cast<MemRefType>(op.getSource().getType());
    auto filterType = dyn_cast<RankedTensorType>(op.getFilter().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!memRefType || !filterType || !resultType || !memRefType.hasStaticShape() ||
        !filterType.hasStaticShape() || !resultType.hasStaticShape())
      return failure();

    if (memRefType.getRank() != 4 || filterType.getRank() != 4 ||
        resultType.getRank() != 4)
      return failure();

    FailureOr<int64_t> idxB = extractConstantInt(op.getIdxB());
    FailureOr<int64_t> idxH = extractConstantInt(op.getIdxH());
    FailureOr<int64_t> idxW = extractConstantInt(op.getIdxW());
    FailureOr<int64_t> idxC = extractConstantInt(op.getIdxC());
    if (failed(idxB) || failed(idxH) || failed(idxW) || failed(idxC))
      return failure();

    ArrayRef<int64_t> pad = op.getPad();
    ArrayRef<int64_t> stride = op.getStride();
    ArrayRef<int64_t> dilation = op.getDilation();
    if (pad.size() != 4 || stride.size() != 2 || dilation.size() != 2)
      return failure();

    FailureOr<Conv2DSpatialSlice> heightSlice = computeConv2DSpatialSlice(
        memRefType.getDimSize(1), *idxH, resultType.getDimSize(1),
        filterType.getDimSize(0), pad[0], pad[2], stride[0], dilation[0]);
    FailureOr<Conv2DSpatialSlice> widthSlice = computeConv2DSpatialSlice(
        memRefType.getDimSize(2), *idxW, resultType.getDimSize(2),
        filterType.getDimSize(1), pad[1], pad[3], stride[1], dilation[1]);
    if (failed(heightSlice) || failed(widthSlice))
      return failure();

    int64_t inputChannels = filterType.getDimSize(2);
    int64_t outputChannels = resultType.getDimSize(3);
    if (inputChannels <= 0 || outputChannels != filterType.getDimSize(3))
      return failure();

    SmallVector<int64_t> loadStart = {
        *idxB * resultType.getDimSize(0),
        heightSlice->loadStart,
        widthSlice->loadStart,
        *idxC * inputChannels,
    };
    SmallVector<int64_t> loadShape = {
        resultType.getDimSize(0),
        heightSlice->loadSize,
        widthSlice->loadSize,
        inputChannels,
    };
    auto loadType = RankedTensorType::get(loadShape, memRefType.getElementType());

    OperationState loadState(op.getLoc(), "nova.load");
    loadState.addOperands(op.getSource());
    loadState.addTypes(loadType);
    loadState.addAttribute("start",
                           DenseI64ArrayAttr::get(rewriter.getContext(), loadStart));
    Operation *loadOp = rewriter.create(loadState);

    SmallVector<int64_t> convPad = {
        heightSlice->convPadBefore,
        widthSlice->convPadBefore,
        heightSlice->convPadAfter,
        widthSlice->convPadAfter,
    };
    OperationState convState(op.getLoc(), "nova.conv2d");
    convState.addOperands({loadOp->getResult(0), op.getFilter()});
    convState.addTypes(resultType);
    convState.addAttribute("pad",
                           DenseI64ArrayAttr::get(rewriter.getContext(), convPad));
    convState.addAttribute("stride", op.getStrideAttr());
    convState.addAttribute("dilation", op.getDilationAttr());
    if (auto group = op.getGroupAttr())
      convState.addAttribute("group", group);

    Operation *convOp = rewriter.create(convState);
    rewriter.replaceOp(op, convOp->getResults());
    return success();
  }
};

struct StoreOpToNovaPattern : OpRewritePattern<xt::StoreOp> {
  using OpRewritePattern<xt::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xt::StoreOp op,
                                PatternRewriter &rewriter) const override {
    auto valueType = dyn_cast<RankedTensorType>(op.getValue().getType());
    if (!valueType)
      return failure();

    FailureOr<DenseI64ArrayAttr> startAttr = extractConstantStarts(
        rewriter.getContext(), op.getCoords(), valueType);
    if (failed(startAttr))
      return failure();

    OperationState state(op.getLoc(), "nova.store");
    state.addOperands({op.getValue(), op.getDest()});
    state.addAttribute("start", *startAttr);

    Operation *novaOp = rewriter.create(state);
    rewriter.eraseOp(op);
    (void)novaOp;
    return success();
  }
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

template <>
StringRef UnaryTensorOpToNovaPattern<xt::ExpOp>::getNovaName() {
  return "nova.exp";
}

template <>
StringRef UnaryTensorOpToNovaPattern<xt::ReciprocalOp>::getNovaName() {
  return "nova.reciprocal";
}

template <>
StringRef UnaryTensorOpToNovaPattern<xt::RsqrtOp>::getNovaName() {
  return "nova.rsqrt";
}

template <>
StringRef UnaryCastOpToNovaPattern<xt::IToFOp>::getNovaName() {
  return "nova.itof";
}

template <>
StringRef UnaryCastOpToNovaPattern<xt::FToIOp>::getNovaName() {
  return "nova.ftoi";
}

class XTToNovaPass : public mlir::xt::impl::XTToNovaBase<XTToNovaPass> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<BinaryOpToNovaPattern<xt::AddOp>,
                 BinaryOpToNovaPattern<xt::MulOp>,
                 BinaryOpToNovaPattern<xt::SubOp>,
                 LoadOpToNovaPattern,
                 LoadConv2DOpToNovaPattern,
                 MatmulOpToNovaPattern,
                 PermuteOpToNovaPattern,
                 UnaryTensorOpToNovaPattern<xt::ExpOp>,
                 UnaryTensorOpToNovaPattern<xt::ReciprocalOp>,
                 UnaryTensorOpToNovaPattern<xt::RsqrtOp>,
                 UnaryCastOpToNovaPattern<xt::IToFOp>,
                 UnaryCastOpToNovaPattern<xt::FToIOp>,
                 ReduceOpToNovaPattern<xt::ReduceSumOp>,
                 ReduceOpToNovaPattern<xt::ReduceMaxOp>,
                 StoreOpToNovaPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::xt::createXTToNovaPass() {
  return std::make_unique<XTToNovaPass>();
}
