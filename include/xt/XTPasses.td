#ifndef XT_PASS
#define XT_PASS

include "mlir/Pass/PassBase.td"

def XTSerialize : Pass<"xt-serialize", "::mlir::func::FuncOp"> {
  let summary = "Serialize a grid-annotated xt function body";
  let description = [{
    Expands a single-block void function body once per xt.grid coordinate and
    rewrites xt.get_tile_block_id to concrete arith.constant values.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::xt::XTDialect"
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
