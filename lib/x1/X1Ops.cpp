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
