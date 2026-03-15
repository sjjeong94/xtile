#include "xt/XTDialect.h"
#include "xt/XTOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::xt;

#include "xt/XTOpsDialect.cpp.inc"

void XTDialect::initialize() { addOperations<
#define GET_OP_LIST
#include "xt/XTOps.cpp.inc"
      >(); }

LogicalResult XTDialect::verifyOperationAttribute(Operation *op,
                                                  NamedAttribute attr) {
  if (attr.getName() != "xt.grid")
    return success();

  if (!isa<func::FuncOp>(op))
    return op->emitOpError("xt.grid is only valid on func.func operations");

  auto grid = llvm::dyn_cast<DenseI32ArrayAttr>(attr.getValue());
  if (!grid)
    return op->emitOpError("xt.grid must be a dense i32 array attribute");
  if (grid.size() != 3)
    return op->emitOpError("xt.grid must have exactly 3 entries");
  for (int32_t dim : grid.asArrayRef()) {
    if (dim <= 0)
      return op->emitOpError("xt.grid entries must be positive");
  }
  return success();
}
