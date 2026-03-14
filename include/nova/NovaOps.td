#ifndef NOVA_OPS
#define NOVA_OPS

include "nova/NovaDialect.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Nova_RankedTensor : RankedTensorOf<[AnyType]>;

class Nova_Op<string mnemonic, list<Trait> traits = []> :
    Op<NovaDialect, mnemonic, traits>;

class Nova_BinaryTensorOp<string mnemonic, string summaryText> :
    Nova_Op<mnemonic, [Pure]> {
  let summary = summaryText;
  let arguments = (ins Nova_RankedTensor:$lhs,
                       Nova_RankedTensor:$rhs,
                       I32Attr:$mode,
                       F32Attr:$lhs_s,
                       F32Attr:$lhs_b,
                       F32Attr:$rhs_s,
                       F32Attr:$rhs_b);
  let results = (outs Nova_RankedTensor:$result);
}

def Nova_ElementwiseOp : Nova_BinaryTensorOp<
    "elementwise", "hardware tile elementwise binary op">;

def Nova_BroadcastOp : Nova_BinaryTensorOp<
    "broadcast", "hardware tile broadcast binary op">;

def Nova_ReduceOp : Nova_Op<"reduce", [Pure]> {
  let summary = "hardware tile reduction op";
  let arguments = (ins Nova_RankedTensor:$input, I32Attr:$mode);
  let results = (outs Nova_RankedTensor:$result);
}

#endif // NOVA_OPS
