#include "xt/XTDialect.h"
#include "xt/XTOps.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::xt;

#include "xt/XTOpsDialect.cpp.inc"

void XTDialect::initialize() { addOperations<
#define GET_OP_LIST
#include "xt/XTOps.cpp.inc"
      >(); }
