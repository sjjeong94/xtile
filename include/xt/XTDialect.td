#ifndef XT_DIALECT
#define XT_DIALECT

include "mlir/IR/DialectBase.td"

def XTDialect : Dialect {
  let name = "xt";
  let cppNamespace = "::mlir::xt";
  let summary = "xTile tile-level dialect";
  let description = [{
    The xt dialect models tile-level NPU operations over global-memory memrefs
    and local tile tensors.
  }];
}

#endif // XT_DIALECT
