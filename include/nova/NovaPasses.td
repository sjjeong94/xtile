#ifndef NOVA_PASS
#define NOVA_PASS

include "mlir/Pass/PassBase.td"

def NovaOptimize : Pass<"nova-optimize", "::mlir::func::FuncOp"> {
  let summary = "Optimize nova scalar/binary compositions";
  let description = [{
    Folds nova.scalar into nova.elementwise and nova.broadcast by updating
    operand scale/bias attributes when the scalar mode is supported.
  }];
  let dependentDialects = [
    "mlir::nova::NovaDialect"
  ];
}

def NovaAllocate : Pass<"nova-allocate", "::mlir::func::FuncOp"> {
  let summary = "Allocate addresses for eligible nova tensor values";
  let description = [{
    Computes block-local tensor lifetimes, assigns first-fit addresses, and
    writes each allocated address into the result tensor encoding as a
    DictionaryAttr.
  }];
  let dependentDialects = [
    "mlir::nova::NovaDialect"
  ];
}

#endif // NOVA_PASS
