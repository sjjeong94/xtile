#include "x1/X1Dialect.h"
#include "x1/X1Ops.h"

using namespace mlir;
using namespace mlir::x1;

#include "x1/X1OpsDialect.cpp.inc"

void X1Dialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "x1/X1Ops.cpp.inc"
      >();
}
