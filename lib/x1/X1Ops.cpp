#include "x1/X1Ops.h"

using namespace mlir;
using namespace mlir::x1;

static LogicalResult verifyMemRefCommandBank(Operation *op, IntegerAttr bank) {
  if (bank.getInt() < 0)
    return op->emitOpError("bank attribute must be non-negative");
  return success();
}

static LogicalResult verifyMemRefCommandSpace(Operation *op, IntegerAttr space) {
  if (space.getInt() < 0)
    return op->emitOpError("space attribute must be non-negative");
  return success();
}

static LogicalResult verifyNonNegativeBank(Operation *op, IntegerAttr bank,
                                           StringRef name) {
  if (bank.getInt() < 0)
    return op->emitOpError() << name << " bank attribute must be non-negative";
  return success();
}

static LogicalResult verifyPositiveIntAttr(Operation *op, IntegerAttr attr,
                                           StringRef name) {
  if (attr.getInt() <= 0)
    return op->emitOpError() << name << " attribute must be positive";
  return success();
}

static LogicalResult verifyPositiveArray(Operation *op, DenseI64ArrayAttr attr,
                                         StringRef name, size_t expectedSize) {
  ArrayRef<int64_t> values = attr.asArrayRef();
  if (values.size() != expectedSize)
    return op->emitOpError()
           << name << " attribute must have " << expectedSize << " entries";
  for (int64_t value : values) {
    if (value <= 0)
      return op->emitOpError() << name << " attribute values must be positive";
  }
  return success();
}

LogicalResult Conv2DOp::verify() {
  if (failed(verifyNonNegativeBank(*this, getInpAttr(), "input")))
    return failure();
  if (failed(verifyNonNegativeBank(*this, getFilterAttr(), "filter")))
    return failure();
  if (failed(verifyNonNegativeBank(*this, getOutAttr(), "output")))
    return failure();
  if (failed(verifyPositiveArray(*this, getInputShapeAttr(), "inputShape", 4)))
    return failure();
  if (failed(verifyPositiveArray(*this, getKernelShapeAttr(), "kernelShape", 4)))
    return failure();
  if (failed(verifyPositiveArray(*this, getResultShapeAttr(), "resultShape", 4)))
    return failure();
  if (failed(verifyPositiveIntAttr(*this, getGroupAttr(), "group")))
    return failure();
  if (getPadAttr().asArrayRef().size() != 4)
    return emitOpError("pad attribute must have 4 entries");
  if (failed(verifyPositiveArray(*this, getStrideAttr(), "stride", 2)))
    return failure();
  if (failed(verifyPositiveArray(*this, getDilationAttr(), "dilation", 2)))
    return failure();
  return success();
}

LogicalResult LoadOp::verify() {
  if (failed(verifyMemRefCommandBank(*this, getBankAttr())))
    return failure();
  return verifyMemRefCommandSpace(*this, getSpaceAttr());
}

LogicalResult StoreOp::verify() {
  if (failed(verifyMemRefCommandBank(*this, getBankAttr())))
    return failure();
  return verifyMemRefCommandSpace(*this, getSpaceAttr());
}

#define GET_OP_CLASSES
#include "x1/X1Ops.cpp.inc"
