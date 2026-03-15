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

def NovaSerialize : Pass<"nova-serialize", "::mlir::func::FuncOp"> {
  let summary = "Serialize a grid-annotated xt function body";
  let description = [{
    Expands a single-block void function body once per xt.grid coordinate and
    rewrites xt.get_tile_block_id to concrete arith.constant values.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::nova::NovaDialect",
    "mlir::xt::XTDialect"
  ];
}

#endif // NOVA_PASS
