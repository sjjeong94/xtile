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
  let assemblyFormat = "$mode attr-dict";
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
  let assemblyFormat = "$source $bank $start $shape `space` $space `thread` $thread attr-dict `:` type($source)";
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
  let assemblyFormat = "`lhs` $lhs0 $lhs1 `rhs` $rhs0 $rhs1 `out` $out0 $out1 `mnk` $m $n $k attr-dict";
}

class X1_UnaryBankCommandOp<string mnemonic, string summaryText> :
    X1_Op<mnemonic> {
  let summary = summaryText;
  let arguments = (ins I64Attr:$inp0,
                       I64Attr:$inp1,
                       I64Attr:$out0,
                       I64Attr:$out1,
                       DenseI64ArrayAttr:$shape);
  let assemblyFormat = "`inp` $inp0 $inp1 `out` $out0 $out1 $shape attr-dict";
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
  let assemblyFormat = "$mode `inp` $inp0 $inp1 `out` $out0 $out1 $shape `axis` $axis attr-dict";
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
  let assemblyFormat = "$mode `lhs` $lhs0 $lhs1 `rhs` $rhs0 $rhs1 `out` $out0 $out1 $shape `axis` $axis `lhs` $lhs_a $lhs_b `rhs` $rhs_a $rhs_b attr-dict";
}

def X1_ElementwiseOp : X1_Op<"elementwise"> {
  let summary = "x1 elementwise command";
  let arguments = (ins I64Attr:$lhs0,
                       I64Attr:$lhs1,
                       I64Attr:$rhs0,
                       I64Attr:$rhs1,
                       I64Attr:$out0,
                       I64Attr:$out1,
                       DenseI64ArrayAttr:$shape,
                       F32Attr:$lhs_a,
                       F32Attr:$lhs_b,
                       F32Attr:$rhs_a,
                       F32Attr:$rhs_b,
                       I32Attr:$mode);
  let assemblyFormat = "$mode `lhs` $lhs0 $lhs1 `rhs` $rhs0 $rhs1 `out` $out0 $out1 $shape `lhs` $lhs_a $lhs_b `rhs` $rhs_a $rhs_b attr-dict";
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
  let assemblyFormat = "`inp` $inp0 $inp1 `out` $out0 $out1 $shape attr-dict";
}

def X1_RsqrtOp : X1_UnaryBankCommandOp<
    "rsqrt", "x1 rsqrt command">;

def X1_ReciprocalOp : X1_UnaryBankCommandOp<
    "reciprocal", "x1 reciprocal command">;

#endif // X1_OPS
