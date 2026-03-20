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

def NovaThreading : Pass<"nova-threading", "::mlir::func::FuncOp"> {
  let summary = "Assign and propagate threading in nova tensor encodings";
  let description = [{
    Adds a threading entry to nova.load result encodings based on the output
    tile height and propagates existing threading values through supported
    unary tensor and unary cast ops.
  }];
  let dependentDialects = [
    "mlir::nova::NovaDialect"
  ];
}

def NovaBarrier : Pass<"nova-barrier", "::mlir::func::FuncOp"> {
  let summary = "Insert nova.barrier ops around executable nova compute ops";
  let description = [{
    Inserts nova.barrier(mode = 0) after nova compute ops other than load,
    store, free, and barrier, and inserts nova.barrier(mode = 1) before each
    func.return. Existing matching barriers are preserved so the pass is
    idempotent.
  }];
  let dependentDialects = [
    "mlir::nova::NovaDialect"
  ];
}

def NovaToX1 : Pass<"nova-to-x1", "::mlir::func::FuncOp"> {
  let summary = "Lower supported nova operations into x1 operations";
  let description = [{
    Rewrites the subset of nova operations that already have x1 equivalents.
    This change starts with nova.barrier -> x1.barrier.
  }];
  let dependentDialects = [
    "mlir::nova::NovaDialect",
    "mlir::x1::X1Dialect"
  ];
}

#endif // NOVA_PASS
