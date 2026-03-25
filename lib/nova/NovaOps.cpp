#include "nova/NovaOps.h"

#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::nova;

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

static LogicalResult verifyTensorCastTypes(Operation *op, Value input,
                                           Value result, bool intToFloat) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto resultType = dyn_cast<RankedTensorType>(result.getType());
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

static FailureOr<int64_t> computeConvOutputDim(int64_t inputSize,
                                               int64_t kernelSize,
                                               int64_t padBefore,
                                               int64_t padAfter,
                                               int64_t stride,
                                               int64_t dilation) {
  if (inputSize <= 0 || kernelSize <= 0 || stride <= 0 || dilation <= 0)
    return failure();
  int64_t effectiveKernel = dilation * (kernelSize - 1) + 1;
  int64_t numerator = inputSize + padBefore + padAfter - effectiveKernel;
  if (numerator < 0)
    return failure();
  return numerator / stride + 1;
}

LogicalResult LoadOp::verify() {
  auto tensorType = dyn_cast<RankedTensorType>(getResult().getType());
  auto memRefType = dyn_cast<MemRefType>(getSource().getType());
  if (failed(verifyMemRefAndTensor(*this, memRefType, tensorType)))
    return failure();
  if (static_cast<int64_t>(getStart().size()) != tensorType.getRank())
    return emitOpError("start attribute count must match tensor rank");
  if (auto shared = getSharedAttr();
      shared && shared.getInt() != 0 && shared.getInt() != 1 &&
      shared.getInt() != 2)
    return emitOpError("shared attribute must be 0, 1, or 2");
  return success();
}

LogicalResult StoreOp::verify() {
  auto tensorType = dyn_cast<RankedTensorType>(getValue().getType());
  auto memRefType = dyn_cast<MemRefType>(getDest().getType());
  if (failed(verifyMemRefAndTensor(*this, memRefType, tensorType)))
    return failure();
  if (static_cast<int64_t>(getStart().size()) != tensorType.getRank())
    return emitOpError("start attribute count must match tensor rank");
  if (auto shared = getSharedAttr();
      shared && shared.getInt() != 0 && shared.getInt() != 1 &&
      shared.getInt() != 2)
    return emitOpError("shared attribute must be 0, 1, or 2");
  return success();
}

LogicalResult ReduceOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto resultType = dyn_cast<RankedTensorType>(getResult().getType());
  if (!inputType || !resultType)
    return emitOpError("requires ranked tensor operand and result");
  if (!inputType.hasStaticShape() || !resultType.hasStaticShape())
    return emitOpError("requires statically shaped tensors");
  if (inputType.getElementType() != resultType.getElementType())
    return emitOpError("requires operand and result element types to match");
  if (inputType.getRank() != 2 || resultType.getRank() != 2)
    return emitOpError("requires rank-2 tensors");
  int64_t axis = getAxis();
  if (axis < 0 || axis >= inputType.getRank())
    return emitOpError("axis must be 0 or 1");
  for (int64_t i = 0; i < inputType.getRank(); ++i) {
    int64_t expectedDim = i == axis ? 1 : inputType.getDimSize(i);
    if (resultType.getDimSize(i) != expectedDim)
      return emitOpError(
          "reduce result shape must match input shape except for the reduced dimension, which must be 1");
  }
  return success();
}

LogicalResult Conv2DOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto filterType = dyn_cast<RankedTensorType>(getFilter().getType());
  auto resultType = dyn_cast<RankedTensorType>(getResult().getType());
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
  if (auto group = getGroupAttr(); group && group.getInt() <= 0)
    return emitOpError("group attribute must be positive");
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
      inputType.getDimSize(1), filterType.getDimSize(0), getPadAttr()[0],
      getPadAttr()[2], getStrideAttr()[0], getDilationAttr()[0]);
  FailureOr<int64_t> outW = computeConvOutputDim(
      inputType.getDimSize(2), filterType.getDimSize(1), getPadAttr()[1],
      getPadAttr()[3], getStrideAttr()[1], getDilationAttr()[1]);
  if (failed(outH) || failed(outW))
    return emitOpError("conv2d kernel configuration produces an invalid output shape");
  if (*outH != resultType.getDimSize(1) || *outW != resultType.getDimSize(2))
    return emitOpError("conv2d result spatial dimensions do not match pad/stride/dilation");
  return success();
}

LogicalResult PermuteOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto resultType = dyn_cast<RankedTensorType>(getResult().getType());
  if (!inputType || !resultType)
    return emitOpError("requires ranked tensor operand and result");
  if (!inputType.hasStaticShape() || !resultType.hasStaticShape())
    return emitOpError("requires statically shaped tensors");
  if (inputType.getElementType() != resultType.getElementType())
    return emitOpError("requires operand and result element types to match");
  if (inputType.getRank() != resultType.getRank())
    return emitOpError("requires operand and result ranks to match");

  int64_t rank = inputType.getRank();
  ArrayRef<int64_t> permutation = getPermutation();
  if (static_cast<int64_t>(permutation.size()) != rank)
    return emitOpError("permutation attribute length must match tensor rank");

  SmallVector<bool> seen(rank, false);
  for (int64_t dim : permutation) {
    if (dim < 0 || dim >= rank)
      return emitOpError("permutation entries must be in range [0, rank)");
    if (seen[dim])
      return emitOpError(
          "permutation attribute must contain each dimension exactly once");
    seen[dim] = true;
  }

  for (auto [resultDim, inputIndex] : llvm::zip_equal(resultType.getShape(), permutation)) {
    if (resultDim != inputType.getDimSize(inputIndex))
      return emitOpError(
          "permute result shape must match the input shape reordered by permutation");
  }
  return success();
}

LogicalResult IToFOp::verify() {
  return verifyTensorCastTypes(*this, getInput(), getResult(),
                               /*intToFloat=*/true);
}

LogicalResult FToIOp::verify() {
  return verifyTensorCastTypes(*this, getInput(), getResult(),
                               /*intToFloat=*/false);
}

#define GET_OP_CLASSES
#include "nova/NovaOps.cpp.inc"
