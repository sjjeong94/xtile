#ifndef XT_PASS
#define XT_PASS

include "mlir/Pass/PassBase.td"

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
