#ifndef XT_PASS
#define XT_PASS

include "mlir/Pass/PassBase.td"

def XTLowerToLoops : Pass<"xt-lower-to-loops", "::mlir::func::FuncOp"> {
  let summary = "Lower xt tile ops to standard loop-based MLIR";
  let description = [{
    Rewrites xt tile operations into scf, tensor, memref, arith, and math
    dialects using explicit nested loops.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::math::MathDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect"
  ];
  let options = [
    Option<"blockIdX", "block-id-x", "int32_t", "0",
           "Constant x block id used during lowering">,
    Option<"blockIdY", "block-id-y", "int32_t", "0",
           "Constant y block id used during lowering">,
    Option<"blockIdZ", "block-id-z", "int32_t", "0",
           "Constant z block id used during lowering">
  ];
}

def XTToNova : Pass<"xt-to-nova", "::mlir::func::FuncOp"> {
  let summary = "Rewrite xt binary tensor ops into nova hardware ops";
  let description = [{
    Rewrites xt.add, xt.mul, and xt.sub into nova.elementwise or
    nova.broadcast depending on operand/result tensor shapes.
  }];
  let dependentDialects = [
    "mlir::nova::NovaDialect"
  ];
}

#endif // XT_PASS
