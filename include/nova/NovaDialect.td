#ifndef NOVA_DIALECT
#define NOVA_DIALECT

include "mlir/IR/DialectBase.td"

def NovaDialect : Dialect {
  let name = "nova";
  let cppNamespace = "::mlir::nova";
  let summary = "Nova hardware-oriented tile dialect";
  let description = [{
    The nova dialect models hardware-oriented tile operations produced from xt.
  }];
}

#endif // NOVA_DIALECT
