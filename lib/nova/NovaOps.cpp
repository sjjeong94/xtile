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
