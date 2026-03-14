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

#endif // NOVA_PASS
