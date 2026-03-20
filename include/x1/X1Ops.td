#ifndef X1_OPS
#define X1_OPS

include "x1/X1Dialect.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def X1_RankedMemRef : MemRefOf<[AnyType]>;

class X1_Op<string mnemonic, list<Trait> traits = []> :
    Op<X1Dialect, mnemonic, traits>;

def X1_BarrierOp : X1_Op<"barrier"> {
  let summary = "x1 barrier op";
  let arguments = (ins I32Attr:$mode);
  let assemblyFormat = "`(` `)` attr-dict";
}

class X1_MemRefCommandOp<string mnemonic, string summaryText> :
    X1_Op<mnemonic> {
  let summary = summaryText;
  let arguments = (ins X1_RankedMemRef:$source,
                       I64Attr:$bank,
                       I64Attr:$space,
                       I64Attr:$thread,
                       I64ArrayAttr:$start,
                       I64ArrayAttr:$shape);
  let hasVerifier = 1;
  let assemblyFormat = "`(` $source `)` attr-dict `:` type($source)";
}

def X1_LoadOp : X1_MemRefCommandOp<
    "load", "x1 memory load command">;

def X1_StoreOp : X1_MemRefCommandOp<
    "store", "x1 memory store command">;

def X1_MatmulOp : X1_Op<"matmul"> {
  let summary = "x1 matmul command";
  let arguments = (ins I64Attr:$lhs0_bank,
                       I64Attr:$lhs1_bank,
                       I64Attr:$rhs0_bank,
                       I64Attr:$rhs1_bank,
                       I64Attr:$res0_bank,
                       I64Attr:$res1_bank,
                       I64Attr:$m,
                       I64Attr:$n,
                       I64Attr:$k);
  let assemblyFormat = "`(` `)` attr-dict";
}

class X1_UnaryBankCommandOp<string mnemonic, string summaryText> :
    X1_Op<mnemonic> {
  let summary = summaryText;
  let arguments = (ins I64Attr:$inp0,
                       I64Attr:$inp1,
                       I64Attr:$res0,
                       I64Attr:$res1,
                       I64Attr:$m,
                       I64Attr:$n);
  let assemblyFormat = "`(` `)` attr-dict";
}

def X1_ReduceOp : X1_Op<"reduce"> {
  let summary = "x1 reduce command";
  let arguments = (ins I64Attr:$inp0,
                       I64Attr:$inp1,
                       I64Attr:$res0,
                       I64Attr:$res1,
                       I64Attr:$m,
                       I64Attr:$n,
                       I32Attr:$mode);
  let assemblyFormat = "`(` `)` attr-dict";
}

def X1_BroadcastOp : X1_Op<"broadcast"> {
  let summary = "x1 broadcast command";
  let arguments = (ins I64Attr:$lhs0,
                       I64Attr:$lhs1,
                       I64Attr:$rhs0,
                       I64Attr:$rhs1,
                       I64Attr:$res0,
                       I64Attr:$res1,
                       F32Attr:$lhs_a,
                       F32Attr:$lhs_b,
                       I32Attr:$mode,
                       F32Attr:$rhs_a,
                       F32Attr:$rhs_b);
  let assemblyFormat = "`(` `)` attr-dict";
}

def X1_ExpOp : X1_UnaryBankCommandOp<
    "exp", "x1 exp command">;

def X1_ReciprocalOp : X1_UnaryBankCommandOp<
    "reciprocal", "x1 reciprocal command">;

#endif // X1_OPS
