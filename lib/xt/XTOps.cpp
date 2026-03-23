#include "xt/XTOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::xt;

#define GET_OP_CLASSES
#include "xt/XTOps.cpp.inc"

static LogicalResult verifyMemRefAndTensor(Operation *op, MemRefType memRefType,
                                           RankedTensorType tensorType) {
  if (!memRefType)
    return op->emitOpError("requires a ranked memref operand");
  if (!tensorType || !tensorType.hasStaticShape())
    return op->emitOpError("requires a statically shaped tensor");
  if (memRefType.getRank() != tensorType.getRank())
    return op->emitOpError("memref rank must match tensor rank");
  if (memRefType.getElementType() != tensorType.getElementType())
    return op->emitOpError("requires matching memref/tensor element types");
  return success();
}

LogicalResult LoadOp::verify() {
  auto tensorType = llvm::dyn_cast<RankedTensorType>(getResult().getType());
  auto memRefType = llvm::dyn_cast<MemRefType>(getSource().getType());
  if (!tensorType)
    return emitOpError("requires a ranked tensor result");
  if (getCoords().empty())
    return emitOpError("requires at least one coordinate");
  if (auto shared = getSharedAttr();
      shared && shared.getInt() != 0 && shared.getInt() != 1 &&
      shared.getInt() != 2)
    return emitOpError("shared attribute must be 0, 1, or 2");
  if (static_cast<int64_t>(getCoords().size()) != tensorType.getRank())
    return emitOpError("coordinate count must match tensor rank");
  return verifyMemRefAndTensor(*this, memRefType, tensorType);
}

LogicalResult StoreOp::verify() {
  auto tensorType = llvm::dyn_cast<RankedTensorType>(getValue().getType());
  auto memRefType = llvm::dyn_cast<MemRefType>(getDest().getType());
  if (!tensorType)
    return emitOpError("requires a ranked tensor operand");
  if (getCoords().empty())
    return emitOpError("requires at least one coordinate");
  if (static_cast<int64_t>(getCoords().size()) != tensorType.getRank())
    return emitOpError("coordinate count must match tensor rank");
  return verifyMemRefAndTensor(*this, memRefType, tensorType);
}

static FailureOr<SmallVector<int64_t>> computeBroadcastShape(RankedTensorType lhsType,
                                                             RankedTensorType rhsType) {
  if (!lhsType || !rhsType || !lhsType.hasStaticShape() || !rhsType.hasStaticShape())
    return failure();
  int64_t resultRank = std::max(lhsType.getRank(), rhsType.getRank());
  SmallVector<int64_t> shape(resultRank, 1);
  for (int64_t i = 0; i < resultRank; ++i) {
    int64_t lhsIndex = lhsType.getRank() - 1 - i;
    int64_t rhsIndex = rhsType.getRank() - 1 - i;
    int64_t lhsDim = lhsIndex >= 0 ? lhsType.getDimSize(lhsIndex) : 1;
    int64_t rhsDim = rhsIndex >= 0 ? rhsType.getDimSize(rhsIndex) : 1;
    if (lhsDim != rhsDim && lhsDim != 1 && rhsDim != 1)
      return failure();
    shape[resultRank - 1 - i] = std::max(lhsDim, rhsDim);
  }
  return shape;
}

static LogicalResult verifyBroadcastableTensorTypes(Operation *op, Value lhs, Value rhs,
                                                    Value result) {
  auto lhsType = llvm::dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = llvm::dyn_cast<RankedTensorType>(rhs.getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(result.getType());
  if (!lhsType || !rhsType || !resultType)
    return op->emitOpError("requires ranked tensor operands and result");
  if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() || !resultType.hasStaticShape())
    return op->emitOpError("requires statically shaped tensors");
  if (lhsType.getElementType() != rhsType.getElementType() ||
      lhsType.getElementType() != resultType.getElementType())
    return op->emitOpError("requires operand and result element types to match");
  FailureOr<SmallVector<int64_t>> broadcastShape = computeBroadcastShape(lhsType, rhsType);
  if (failed(broadcastShape))
    return op->emitOpError("operands are not broadcast-compatible with result tensor type");
  if (static_cast<int64_t>(broadcastShape->size()) != resultType.getRank())
    return op->emitOpError("operands are not broadcast-compatible with result tensor type");
  for (auto [expected, actual] : llvm::zip_equal(*broadcastShape, resultType.getShape())) {
    if (expected != actual)
      return op->emitOpError("operands are not broadcast-compatible with result tensor type");
  }
  return success();
}

static LogicalResult verifySameUnaryTensorTypes(Operation *op, Value input,
                                                Value result) {
  auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(result.getType());
  if (!inputType || !resultType)
    return op->emitOpError("requires ranked tensor operand and result");
  if (inputType != resultType)
    return op->emitOpError("requires operand and result tensor types to match");
  if (!inputType.hasStaticShape())
    return op->emitOpError("requires statically shaped tensors");
  return success();
}

static LogicalResult verifyTensorCastTypes(Operation *op, Value input,
                                           Value result, bool intToFloat) {
  auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(result.getType());
  if (!inputType || !resultType)
    return op->emitOpError("requires ranked tensor operand and result");
  if (!inputType.hasStaticShape() || !resultType.hasStaticShape())
    return op->emitOpError("requires statically shaped tensors");
  if (inputType.getShape() != resultType.getShape())
    return op->emitOpError("requires operand and result tensor shapes to match");

  Type inputElem = inputType.getElementType();
  Type resultElem = resultType.getElementType();
  if (intToFloat) {
    if (!llvm::isa<IntegerType>(inputElem) || !llvm::isa<FloatType>(resultElem))
      return op->emitOpError(
          "requires integer input and floating-point result element types");
  } else {
    if (!llvm::isa<FloatType>(inputElem) || !llvm::isa<IntegerType>(resultElem))
      return op->emitOpError(
          "requires floating-point input and integer result element types");
  }
  return success();
}

LogicalResult AddOp::verify() {
  return verifyBroadcastableTensorTypes(*this, getLhs(), getRhs(), getResult());
}

LogicalResult SubOp::verify() {
  return verifyBroadcastableTensorTypes(*this, getLhs(), getRhs(), getResult());
}

LogicalResult MulOp::verify() {
  return verifyBroadcastableTensorTypes(*this, getLhs(), getRhs(), getResult());
}

LogicalResult ExpOp::verify() {
  return verifySameUnaryTensorTypes(*this, getInput(), getResult());
}

LogicalResult IToFOp::verify() {
  return verifyTensorCastTypes(*this, getInput(), getResult(),
                               /*intToFloat=*/true);
}

LogicalResult FToIOp::verify() {
  return verifyTensorCastTypes(*this, getInput(), getResult(),
                               /*intToFloat=*/false);
}

LogicalResult CosOp::verify() { return verifySameUnaryTensorTypes(*this, getInput(), getResult()); }
LogicalResult SinOp::verify() { return verifySameUnaryTensorTypes(*this, getInput(), getResult()); }
LogicalResult ReciprocalOp::verify() { return verifySameUnaryTensorTypes(*this, getInput(), getResult()); }
LogicalResult RsqrtOp::verify() { return verifySameUnaryTensorTypes(*this, getInput(), getResult()); }
LogicalResult SigmoidOp::verify() { return verifySameUnaryTensorTypes(*this, getInput(), getResult()); }
LogicalResult TanhOp::verify() { return verifySameUnaryTensorTypes(*this, getInput(), getResult()); }
LogicalResult SiluOp::verify() { return verifySameUnaryTensorTypes(*this, getInput(), getResult()); }

static LogicalResult verifyMatmulLikeShape(Operation *op, RankedTensorType lhsType,
                                           RankedTensorType rhsType,
                                           RankedTensorType resultType) {
  if (!lhsType || !rhsType || !resultType)
    return op->emitOpError("requires ranked tensor operands and result");
  if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
      !resultType.hasStaticShape())
    return op->emitOpError("requires statically shaped tensors");
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 || resultType.getRank() != 2)
    return op->emitOpError("requires rank-2 tensors");
  if (lhsType.getDimSize(1) != rhsType.getDimSize(0))
    return op->emitOpError(
        "matmul requires lhs inner dimension to match rhs outer dimension");
  if (resultType.getDimSize(0) != lhsType.getDimSize(0) ||
      resultType.getDimSize(1) != rhsType.getDimSize(1))
    return op->emitOpError("result shape must match matmul output shape");
  return success();
}

static FailureOr<int64_t> computeConvOutputDim(int64_t inputSize, int64_t kernelSize,
                                               int64_t padBefore, int64_t padAfter,
                                               int64_t stride, int64_t dilation) {
  int64_t effectiveKernel = dilation * (kernelSize - 1) + 1;
  int64_t numerator = inputSize + padBefore + padAfter - effectiveKernel;
  if (numerator < 0)
    return failure();
  return numerator / stride + 1;
}

LogicalResult MatmulOp::verify() {
  auto lhsType = llvm::dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsType = llvm::dyn_cast<RankedTensorType>(getRhs().getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(getResult().getType());
  if (failed(verifyMatmulLikeShape(*this, lhsType, rhsType, resultType)))
    return failure();
  if (lhsType.getElementType() == rhsType.getElementType() &&
      lhsType.getElementType() == resultType.getElementType())
    return success();
  if (lhsType.getElementType().isInteger(8) &&
      rhsType.getElementType().isInteger(8) &&
      (resultType.getElementType().isF32() || resultType.getElementType().isBF16()))
    return success();
  return emitOpError(
      "requires matching element types or i8 inputs with f32/bf16 result");
  return success();
}

LogicalResult MMAOp::verify() {
  auto lhsType = llvm::dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsType = llvm::dyn_cast<RankedTensorType>(getRhs().getType());
  auto accType = llvm::dyn_cast<RankedTensorType>(getAcc().getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(getResult().getType());
  if (failed(verifyMatmulLikeShape(*this, lhsType, rhsType, resultType)))
    return failure();
  if (!accType || !accType.hasStaticShape() || accType.getRank() != 2)
    return emitOpError("requires a rank-2 statically shaped accumulator");
  if (accType != resultType)
    return emitOpError("accumulator and result tensor types must match");
  if (!lhsType.getElementType().isInteger(8) || !rhsType.getElementType().isInteger(8))
    return emitOpError("mma requires i8 input tensors");
  Type accElem = accType.getElementType();
  if (!accElem.isF32() && !accElem.isBF16())
    return emitOpError("mma requires f32 or bf16 accumulator and result tensors");
  return success();
}

LogicalResult Conv2DOp::verify() {
  auto inputType = llvm::dyn_cast<RankedTensorType>(getInput().getType());
  auto filterType = llvm::dyn_cast<RankedTensorType>(getFilter().getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(getResult().getType());
  if (!inputType || !filterType || !resultType)
    return emitOpError("requires ranked tensor operands and result");
  if (!inputType.hasStaticShape() || !filterType.hasStaticShape() ||
      !resultType.hasStaticShape())
    return emitOpError("requires statically shaped tensors");
  if (inputType.getRank() != 4 || filterType.getRank() != 4 ||
      resultType.getRank() != 4)
    return emitOpError("requires rank-4 input, filter, and result tensors");
  if (getPadAttr().size() != 4)
    return emitOpError("pad attribute must have exactly 4 entries");
  if (getStrideAttr().size() != 2)
    return emitOpError("stride attribute must have exactly 2 entries");
  if (getDilationAttr().size() != 2)
    return emitOpError("dilation attribute must have exactly 2 entries");
  for (int64_t pad : getPadAttr().asArrayRef()) {
    if (pad < 0)
      return emitOpError("pad attribute entries must be non-negative");
  }
  for (int64_t stride : getStrideAttr().asArrayRef()) {
    if (stride <= 0)
      return emitOpError("stride attribute entries must be positive");
  }
  for (int64_t dilation : getDilationAttr().asArrayRef()) {
    if (dilation <= 0)
      return emitOpError("dilation attribute entries must be positive");
  }
  if (!inputType.getElementType().isInteger(8) ||
      !filterType.getElementType().isInteger(8))
    return emitOpError("conv2d requires i8 input and filter tensors");
  if (!resultType.getElementType().isF32() &&
      !resultType.getElementType().isBF16())
    return emitOpError("conv2d requires f32 or bf16 result tensors");
  if (inputType.getDimSize(3) != filterType.getDimSize(2))
    return emitOpError("conv2d requires input and filter channel dimensions to match");
  if (inputType.getDimSize(0) != resultType.getDimSize(0))
    return emitOpError("conv2d result batch dimension must match input");
  if (filterType.getDimSize(3) != resultType.getDimSize(3))
    return emitOpError("conv2d result channel dimension must match filter output channels");

  FailureOr<int64_t> outH = computeConvOutputDim(
      inputType.getDimSize(1), filterType.getDimSize(0), getPadAttr()[0], getPadAttr()[2],
      getStrideAttr()[0], getDilationAttr()[0]);
  FailureOr<int64_t> outW = computeConvOutputDim(
      inputType.getDimSize(2), filterType.getDimSize(1), getPadAttr()[1], getPadAttr()[3],
      getStrideAttr()[1], getDilationAttr()[1]);
  if (failed(outH) || failed(outW))
    return emitOpError("conv2d kernel configuration produces an invalid output shape");
  if (*outH != resultType.getDimSize(1) || *outW != resultType.getDimSize(2))
    return emitOpError("conv2d result spatial dimensions do not match pad/stride/dilation");
  return success();
}

LogicalResult DepthwiseConv2DOp::verify() {
  auto inputType = llvm::dyn_cast<RankedTensorType>(getInput().getType());
  auto filterType = llvm::dyn_cast<RankedTensorType>(getFilter().getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(getResult().getType());
  if (!inputType || !filterType || !resultType)
    return emitOpError("requires ranked tensor operands and result");
  if (!inputType.hasStaticShape() || !filterType.hasStaticShape() ||
      !resultType.hasStaticShape())
    return emitOpError("requires statically shaped tensors");
  if (inputType.getRank() != 4 || filterType.getRank() != 4 ||
      resultType.getRank() != 4)
    return emitOpError("requires rank-4 input, filter, and result tensors");
  if (getPadAttr().size() != 4)
    return emitOpError("pad attribute must have exactly 4 entries");
  if (getStrideAttr().size() != 2)
    return emitOpError("stride attribute must have exactly 2 entries");
  if (getDilationAttr().size() != 2)
    return emitOpError("dilation attribute must have exactly 2 entries");
  for (int64_t pad : getPadAttr().asArrayRef()) {
    if (pad < 0)
      return emitOpError("pad attribute entries must be non-negative");
  }
  for (int64_t stride : getStrideAttr().asArrayRef()) {
    if (stride <= 0)
      return emitOpError("stride attribute entries must be positive");
  }
  for (int64_t dilation : getDilationAttr().asArrayRef()) {
    if (dilation <= 0)
      return emitOpError("dilation attribute entries must be positive");
  }
  if (!inputType.getElementType().isInteger(8) ||
      !filterType.getElementType().isInteger(8))
    return emitOpError("depthwise_conv2d requires i8 input and filter tensors");
  if (!resultType.getElementType().isF32() &&
      !resultType.getElementType().isBF16())
    return emitOpError("depthwise_conv2d requires f32 or bf16 result tensors");
  if (filterType.getDimSize(2) != 1)
    return emitOpError("depthwise_conv2d requires filter input-channel dimension to be 1");
  if (inputType.getDimSize(0) != resultType.getDimSize(0))
    return emitOpError("depthwise_conv2d result batch dimension must match input");
  if (inputType.getDimSize(3) != resultType.getDimSize(3))
    return emitOpError("depthwise_conv2d requires input and result channel dimensions to match");
  if (filterType.getDimSize(3) != inputType.getDimSize(3))
    return emitOpError("depthwise_conv2d requires filter channel dimension to match input/output channels");

  FailureOr<int64_t> outH = computeConvOutputDim(
      inputType.getDimSize(1), filterType.getDimSize(0), getPadAttr()[0], getPadAttr()[2],
      getStrideAttr()[0], getDilationAttr()[0]);
  FailureOr<int64_t> outW = computeConvOutputDim(
      inputType.getDimSize(2), filterType.getDimSize(1), getPadAttr()[1], getPadAttr()[3],
      getStrideAttr()[1], getDilationAttr()[1]);
  if (failed(outH) || failed(outW))
    return emitOpError("depthwise_conv2d kernel configuration produces an invalid output shape");
  if (*outH != resultType.getDimSize(1) || *outW != resultType.getDimSize(2))
    return emitOpError("depthwise_conv2d result spatial dimensions do not match pad/stride/dilation");
  return success();
}

static LogicalResult verifyLastDimReductionShape(Operation *op, Value input,
                                                 Value result, int64_t axis) {
  auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(result.getType());
  if (!inputType || !resultType)
    return op->emitOpError("requires ranked tensor operand and result");
  if (!inputType.hasStaticShape() || !resultType.hasStaticShape())
    return op->emitOpError("requires statically shaped tensors");
  if (inputType.getElementType() != resultType.getElementType())
    return op->emitOpError("requires operand and result element types to match");
  if (inputType.getRank() != resultType.getRank())
    return op->emitOpError("requires operand and result ranks to match");
  if (inputType.getRank() != 2)
    return op->emitOpError("reduce currently supports rank-2 tensors only");
  int64_t rank = inputType.getRank();
  if (axis < -rank || axis >= rank)
    return op->emitOpError("axis must be in range [-rank, rank)");
  int64_t normalizedAxis = axis >= 0 ? axis : axis + rank;
  for (int64_t i = 0, e = inputType.getRank(); i < e; ++i) {
    int64_t expectedDim = i == normalizedAxis ? 1 : inputType.getDimSize(i);
    if (resultType.getDimSize(i) != expectedDim)
      return op->emitOpError(
          "reduce result shape must match input shape except for the reduced dimension, which must be 1");
  }
  return success();
}

LogicalResult ReduceSumOp::verify() {
  return verifyLastDimReductionShape(*this, getInput(), getResult(), getAxis());
}

LogicalResult ReduceMaxOp::verify() {
  return verifyLastDimReductionShape(*this, getInput(), getResult(), getAxis());
}

static LogicalResult verifyTransposeShape(Operation *op, Value input,
                                          Value result) {
  auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(result.getType());
  if (!inputType || !resultType)
    return op->emitOpError("requires ranked tensor operand and result");
  if (!inputType.hasStaticShape() || !resultType.hasStaticShape())
    return op->emitOpError("requires statically shaped tensors");
  if (inputType.getElementType() != resultType.getElementType())
    return op->emitOpError("requires operand and result element types to match");
  if (inputType.getRank() != 3 || resultType.getRank() != 3)
    return op->emitOpError("transpose requires rank-3 operand and result tensors");
  if (inputType.getDimSize(0) != resultType.getDimSize(0) ||
      inputType.getDimSize(1) != resultType.getDimSize(2) ||
      inputType.getDimSize(2) != resultType.getDimSize(1))
    return op->emitOpError(
        "transpose result shape must preserve dim 0 and swap dims 1 and 2");
  return success();
}

static LogicalResult verifyPermuteShape(Operation *op, Value input, Value result,
                                        DenseI64ArrayAttr permutationAttr) {
  auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(result.getType());
  if (!inputType || !resultType)
    return op->emitOpError("requires ranked tensor operand and result");
  if (!inputType.hasStaticShape() || !resultType.hasStaticShape())
    return op->emitOpError("requires statically shaped tensors");
  if (inputType.getElementType() != resultType.getElementType())
    return op->emitOpError("requires operand and result element types to match");
  if (inputType.getRank() != resultType.getRank())
    return op->emitOpError("requires operand and result ranks to match");

  int64_t rank = inputType.getRank();
  ArrayRef<int64_t> permutation = permutationAttr.asArrayRef();
  if (static_cast<int64_t>(permutation.size()) != rank)
    return op->emitOpError("permutation attribute length must match tensor rank");

  SmallVector<bool> seen(rank, false);
  for (int64_t dim : permutation) {
    if (dim < 0 || dim >= rank)
      return op->emitOpError("permutation entries must be in range [0, rank)");
    if (seen[dim])
      return op->emitOpError(
          "permutation attribute must contain each dimension exactly once");
    seen[dim] = true;
  }

  for (auto [resultDim, inputIndex] : llvm::zip_equal(resultType.getShape(), permutation)) {
    if (resultDim != inputType.getDimSize(inputIndex))
      return op->emitOpError(
          "permute result shape must match the input shape reordered by permutation");
  }
  return success();
}

static int64_t getStaticElementCount(RankedTensorType type) {
  int64_t count = 1;
  for (int64_t dim : type.getShape())
    count *= dim;
  return count;
}

LogicalResult ReshapeOp::verify() {
  auto inputType = llvm::dyn_cast<RankedTensorType>(getInput().getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(getResult().getType());
  if (!inputType || !resultType)
    return emitOpError("requires ranked tensor operand and result");
  if (!inputType.hasStaticShape() || !resultType.hasStaticShape())
    return emitOpError("requires statically shaped tensors");
  if (inputType.getElementType() != resultType.getElementType())
    return emitOpError("requires operand and result element types to match");
  if (getStaticElementCount(inputType) != getStaticElementCount(resultType))
    return emitOpError(
        "reshape requires operand and result to have the same number of elements");
  return success();
}

LogicalResult TransposeOp::verify() {
  return verifyTransposeShape(*this, getInput(), getResult());
}

LogicalResult PermuteOp::verify() {
  return verifyPermuteShape(*this, getInput(), getResult(), getPermutationAttr());
}

static bool isZeroTensor(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  auto dense = llvm::dyn_cast<DenseElementsAttr>(constant.getValue());
  if (!dense || !dense.isSplat())
    return false;
  auto splat = dense.getSplatValue<Attribute>();
  if (auto floatAttr = llvm::dyn_cast<FloatAttr>(splat))
    return floatAttr.getValue().isZero();
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(splat))
    return intAttr.getValue().isZero();
  return false;
}

namespace {
struct FoldAddZeroPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (isZeroTensor(op.getLhs())) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }
    if (isZeroTensor(op.getRhs())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    return failure();
  }
};

struct FoldSubZeroPattern : public OpRewritePattern<SubOp> {
  using OpRewritePattern<SubOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubOp op,
                                PatternRewriter &rewriter) const override {
    if (!isZeroTensor(op.getRhs()))
      return failure();
    rewriter.replaceOp(op, op.getLhs());
    return success();
  }
};
} // namespace

void AddOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<FoldAddZeroPattern>(context);
}

void SubOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<FoldSubZeroPattern>(context);
}
