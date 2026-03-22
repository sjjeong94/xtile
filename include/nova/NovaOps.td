#ifndef NOVA_OPS
#define NOVA_OPS

include "nova/NovaDialect.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Nova_RankedTensor : RankedTensorOf<[AnyType]>;
def Nova_RankedMemRef : MemRefOf<[AnyType]>;

class Nova_Op<string mnemonic, list<Trait> traits = []> :
    Op<NovaDialect, mnemonic, traits>;

class Nova_UnaryTensorOp<string mnemonic, string summaryText> :
    Nova_Op<mnemonic, [Pure]> {
  let summary = summaryText;
  let arguments = (ins Nova_RankedTensor:$input);
  let results = (outs Nova_RankedTensor:$result);
  let assemblyFormat =
      "`(` $input `)` attr-dict `:` type($input) `->` type($result)";
}

class Nova_UnaryCastOp<string mnemonic, string summaryText> :
    Nova_Op<mnemonic, [Pure]> {
  let summary = summaryText;
  let arguments = (ins Nova_RankedTensor:$input);
  let results = (outs Nova_RankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "`(` $input `)` attr-dict `:` type($input) `->` type($result)";
}

def Nova_BarrierOp : Nova_Op<"barrier"> {
  let summary = "hardware tile barrier op";
  let arguments = (ins I32Attr:$mode);
  let assemblyFormat = "`(` `)` attr-dict";
}

def Nova_LoadOp : Nova_Op<"load"> {
  let summary = "load a tensor tile from a memref using constant start offsets";
  let arguments = (ins Nova_RankedMemRef:$source,
                       I64ArrayAttr:$start,
                       OptionalAttr<I64Attr>:$shared);
  let results = (outs Nova_RankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "`(` $source `)` attr-dict `:` type($source) `->` type($result)";
}

def Nova_StoreOp : Nova_Op<"store"> {
  let summary = "store a tensor tile to a memref using constant start offsets";
  let arguments = (ins Nova_RankedTensor:$value,
                       Nova_RankedMemRef:$dest,
                       I64ArrayAttr:$start,
                       OptionalAttr<I64Attr>:$shared);
  let hasVerifier = 1;
  let assemblyFormat =
      "`(` $value `,` $dest `)` attr-dict `:` functional-type(operands, results)";
}

class Nova_BinaryTensorOp<string mnemonic, string summaryText> :
    Nova_Op<mnemonic, [Pure]> {
  let summary = summaryText;
  let arguments = (ins Nova_RankedTensor:$lhs,
                       Nova_RankedTensor:$rhs,
                       I32Attr:$mode,
                       F32Attr:$lhs_a,
                       F32Attr:$lhs_b,
                       F32Attr:$rhs_a,
                       F32Attr:$rhs_b);
  let results = (outs Nova_RankedTensor:$result);
  let assemblyFormat =
      "`(` $lhs `,` $rhs `)` attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Nova_ElementwiseOp : Nova_BinaryTensorOp<
    "elementwise", "hardware tile elementwise binary op">;

def Nova_BroadcastOp : Nova_BinaryTensorOp<
    "broadcast", "hardware tile broadcast binary op">;

def Nova_ScalarOp : Nova_Op<"scalar", [Pure]> {
  let summary = "hardware tile tensor-scalar binary op";
  let arguments = (ins Nova_RankedTensor:$input, I32Attr:$mode, F32Attr:$rhs);
  let results = (outs Nova_RankedTensor:$result);
  let assemblyFormat =
      "`(` $input `)` attr-dict `:` type($input) `->` type($result)";
}

def Nova_ScalarFmaOp : Nova_Op<"scalar_fma", [Pure]> {
  let summary = "hardware tile tensor scalar fused multiply-add op";
  let arguments = (ins Nova_RankedTensor:$input, F32Attr:$a, F32Attr:$b);
  let results = (outs Nova_RankedTensor:$result);
  let assemblyFormat =
      "`(` $input `)` attr-dict `:` type($input) `->` type($result)";
}

def Nova_SquareOp : Nova_UnaryTensorOp<
    "square", "hardware tile unary square op">;

def Nova_ExpOp : Nova_UnaryTensorOp<
    "exp", "hardware tile unary exp op">;

def Nova_ReciprocalOp : Nova_UnaryTensorOp<
    "reciprocal", "hardware tile unary reciprocal op">;

def Nova_RsqrtOp : Nova_UnaryTensorOp<
    "rsqrt", "hardware tile unary rsqrt op">;

def Nova_IToFOp : Nova_UnaryCastOp<
    "itof", "hardware tile integer-to-float tensor cast op">;

def Nova_FToIOp : Nova_UnaryCastOp<
    "ftoi", "hardware tile float-to-integer tensor cast op">;

def Nova_MatmulOp : Nova_Op<"matmul", [Pure]> {
  let summary = "hardware tile matmul op";
  let arguments = (ins Nova_RankedTensor:$lhs, Nova_RankedTensor:$rhs,
                       Nova_RankedTensor:$scale, Nova_RankedTensor:$bias);
  let results = (outs Nova_RankedTensor:$result);
  let assemblyFormat =
      "`(` $lhs `,` $rhs `,` $scale `,` $bias `)` attr-dict `:` type($lhs) `,` type($rhs) `,` type($scale) `,` type($bias) `->` type($result)";
}

def Nova_ReduceOp : Nova_Op<"reduce", [Pure]> {
  let summary = "hardware tile reduction op";
  let arguments = (ins Nova_RankedTensor:$input, I32Attr:$mode);
  let results = (outs Nova_RankedTensor:$result);
  let assemblyFormat =
      "`(` $input `)` attr-dict `:` type($input) `->` type($result)";
}

#endif // NOVA_OPS
