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

LogicalResult LoadOp::verify() {
  auto tensorType = dyn_cast<RankedTensorType>(getResult().getType());
  auto memRefType = dyn_cast<MemRefType>(getSource().getType());
  if (failed(verifyMemRefAndTensor(*this, memRefType, tensorType)))
    return failure();
  if (static_cast<int64_t>(getIndex().size()) != tensorType.getRank())
    return emitOpError("index attribute count must match tensor rank");
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
  if (static_cast<int64_t>(getIndex().size()) != tensorType.getRank())
    return emitOpError("index attribute count must match tensor rank");
  if (auto shared = getSharedAttr();
      shared && shared.getInt() != 0 && shared.getInt() != 1 &&
      shared.getInt() != 2)
    return emitOpError("shared attribute must be 0, 1, or 2");
  return success();
}

#define GET_OP_CLASSES
#include "nova/NovaOps.cpp.inc"
