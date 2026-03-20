#ifndef X1_DIALECT
#define X1_DIALECT

include "mlir/IR/DialectBase.td"

def X1Dialect : Dialect {
  let name = "x1";
  let cppNamespace = "::mlir::x1";
  let summary = "X1 command dialect";
  let description = [{
    The x1 dialect models low-level command-style hardware operations.
  }];
}

#endif // X1_DIALECT
