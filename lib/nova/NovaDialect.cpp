#include "nova/NovaDialect.h"
#include "nova/NovaOps.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::nova;

#include "nova/NovaOpsDialect.cpp.inc"

void NovaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "nova/NovaOps.cpp.inc"
      >();
}
