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
  let assemblyFormat = "` ` `{` `mode` `=` $mode `}` attr-dict";
}

class X1_MemRefCommandOp<string mnemonic, string summaryText> :
    X1_Op<mnemonic> {
  let summary = summaryText;
  let arguments = (ins X1_RankedMemRef:$source,
                       I64Attr:$bank,
                       I64Attr:$space,
                       I64Attr:$thread,
                       DenseI64ArrayAttr:$start,
                       DenseI64ArrayAttr:$shape);
  let hasVerifier = 1;
  let assemblyFormat = "$source ` ` `{` `bank` `=` $bank `,` `space` `=` $space `,` `thread` `=` $thread `,` `start` `=` $start `,` `shape` `=` $shape `}` attr-dict `:` type($source)";
}

def X1_LoadOp : X1_MemRefCommandOp<
    "load", "x1 memory load command">;

def X1_StoreOp : X1_MemRefCommandOp<
    "store", "x1 memory store command">;

def X1_MatmulOp : X1_Op<"matmul"> {
  let summary = "x1 matmul command";
  let arguments = (ins I64Attr:$lhs0,
                       I64Attr:$lhs1,
                       I64Attr:$rhs0,
                       I64Attr:$rhs1,
                       I64Attr:$out0,
                       I64Attr:$out1,
                       I64Attr:$m,
                       I64Attr:$n,
                       I64Attr:$k);
  let assemblyFormat = "` ` `{` `lhs0` `=` $lhs0 `,` `lhs1` `=` $lhs1 `,` `rhs0` `=` $rhs0 `,` `rhs1` `=` $rhs1 `,` `out0` `=` $out0 `,` `out1` `=` $out1 `,` `m` `=` $m `,` `n` `=` $n `,` `k` `=` $k `}` attr-dict";
}

class X1_UnaryBankCommandOp<string mnemonic, string summaryText> :
    X1_Op<mnemonic> {
  let summary = summaryText;
  let arguments = (ins I64Attr:$inp0,
                       I64Attr:$inp1,
                       I64Attr:$out0,
                       I64Attr:$out1,
                       DenseI64ArrayAttr:$shape);
  let assemblyFormat = "` ` `{` `inp0` `=` $inp0 `,` `inp1` `=` $inp1 `,` `out0` `=` $out0 `,` `out1` `=` $out1 `,` `shape` `=` $shape `}` attr-dict";
}

def X1_ReduceOp : X1_Op<"reduce"> {
  let summary = "x1 reduce command";
  let arguments = (ins I64Attr:$inp0,
                       I64Attr:$inp1,
                       I64Attr:$out0,
                       I64Attr:$out1,
                       DenseI64ArrayAttr:$shape,
                       I32Attr:$axis,
                       I32Attr:$mode);
  let assemblyFormat = "` ` `{` `inp0` `=` $inp0 `,` `inp1` `=` $inp1 `,` `out0` `=` $out0 `,` `out1` `=` $out1 `,` `shape` `=` $shape `,` `axis` `=` $axis `,` `mode` `=` $mode `}` attr-dict";
}

def X1_BroadcastOp : X1_Op<"broadcast"> {
  let summary = "x1 broadcast command";
  let arguments = (ins I64Attr:$lhs0,
                       I64Attr:$lhs1,
                       I64Attr:$rhs0,
                       I64Attr:$rhs1,
                       I64Attr:$out0,
                       I64Attr:$out1,
                       DenseI64ArrayAttr:$shape,
                       I32Attr:$axis,
                       F32Attr:$lhs_a,
                       F32Attr:$lhs_b,
                       F32Attr:$rhs_a,
                       F32Attr:$rhs_b,
                       I32Attr:$mode);
  let assemblyFormat = "` ` `{` `lhs0` `=` $lhs0 `,` `lhs1` `=` $lhs1 `,` `rhs0` `=` $rhs0 `,` `rhs1` `=` $rhs1 `,` `out0` `=` $out0 `,` `out1` `=` $out1 `,` `shape` `=` $shape `,` `axis` `=` $axis `,` `lhs_a` `=` $lhs_a `,` `lhs_b` `=` $lhs_b `,` `rhs_a` `=` $rhs_a `,` `rhs_b` `=` $rhs_b `,` `mode` `=` $mode `}` attr-dict";
}

def X1_ExpOp : X1_UnaryBankCommandOp<
    "exp", "x1 exp command">;

def X1_SquareOp : X1_UnaryBankCommandOp<
    "square", "x1 square command">;

def X1_ScalarFmaOp : X1_Op<"scalar_fma"> {
  let summary = "x1 scalar fma command";
  let arguments = (ins I64Attr:$inp0,
                       I64Attr:$inp1,
                       I64Attr:$out0,
                       I64Attr:$out1,
                       DenseI64ArrayAttr:$shape,
                       F32Attr:$a,
                       F32Attr:$b);
  let assemblyFormat = "` ` `{` `inp0` `=` $inp0 `,` `inp1` `=` $inp1 `,` `out0` `=` $out0 `,` `out1` `=` $out1 `,` `shape` `=` $shape `,` `a` `=` $a `,` `b` `=` $b `}` attr-dict";
}

def X1_RsqrtOp : X1_UnaryBankCommandOp<
    "rsqrt", "x1 rsqrt command">;

def X1_ReciprocalOp : X1_UnaryBankCommandOp<
    "reciprocal", "x1 reciprocal command">;

#endif // X1_OPS
