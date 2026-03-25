#ifndef XT_OPS
#define XT_OPS

include "xt/XTDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
def XT_AnyCoord : AnyTypeOf<[I32, Index]>;
def XT_RankedTensor : RankedTensorOf<[AnyType]>;
def XT_RankedMemRef : MemRefOf<[AnyType]>;

class XT_Op<string mnemonic, list<Trait> traits = []> :
    Op<XTDialect, mnemonic, traits>;

def XT_GetTileBlockIdOp : XT_Op<"get_tile_block_id", [Pure]> {
  let summary = "query the current tile block id";
  let results = (outs I32:$x, I32:$y, I32:$z);
  let assemblyFormat = "attr-dict `:` type($x) `,` type($y) `,` type($z)";
}

def XT_LoadOp : XT_Op<"load"> {
  let summary = "load a tensor tile from a memref";
  let arguments = (ins XT_RankedMemRef:$source,
                       Variadic<XT_AnyCoord>:$coords,
                       OptionalAttr<I64Attr>:$shared);
  let results = (outs XT_RankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "`(` $source `,` $coords `)` attr-dict `:` functional-type(operands, results)";
}

def XT_LoadConv2DOp : XT_Op<"load_conv2d"> {
  let summary = "load a tensor tile from a memref and apply conv2d with a filter tensor";
  let arguments = (ins XT_RankedMemRef:$source,
                       XT_RankedTensor:$filter,
                       Variadic<XT_AnyCoord>:$coords,
                       OptionalAttr<I64Attr>:$group,
                       DenseI64ArrayAttr:$pad,
                       DenseI64ArrayAttr:$stride,
                       DenseI64ArrayAttr:$dilation);
  let results = (outs XT_RankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "`(` $source `,` $filter `,` $coords `)` attr-dict `:` functional-type(operands, results)";
}

def XT_StoreOp : XT_Op<"store"> {
  let summary = "store a tensor tile into a memref";
  let arguments = (ins XT_RankedTensor:$value,
                       XT_RankedMemRef:$dest,
                       Variadic<XT_AnyCoord>:$coords);
  let hasVerifier = 1;
  let assemblyFormat =
      "`(` $value `,` $dest `,` $coords `)` attr-dict `:` functional-type(operands, results)";
}

class XT_BinaryElementwiseOp<string mnemonic, string summaryText> :
    XT_Op<mnemonic, [Pure]> {
  let summary = summaryText;
  let arguments = (ins XT_RankedTensor:$lhs, XT_RankedTensor:$rhs);
  let results = (outs XT_RankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "`(` $lhs `,` $rhs `)` attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def XT_AddOp : XT_BinaryElementwiseOp<"add", "elementwise tile add"> {
  let hasCanonicalizer = 1;
}

def XT_SubOp : XT_BinaryElementwiseOp<"sub", "elementwise tile sub"> {
  let hasCanonicalizer = 1;
}

def XT_MulOp : XT_BinaryElementwiseOp<"mul", "elementwise tile mul">;

class XT_UnaryElementwiseOp<string mnemonic, string summaryText> :
    XT_Op<mnemonic, [Pure]> {
  let summary = summaryText;
  let arguments = (ins XT_RankedTensor:$input);
  let results = (outs XT_RankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "`(` $input `)` attr-dict `:` type($input) `->` type($result)";
}

class XT_UnaryCastOp<string mnemonic, string summaryText> :
    XT_Op<mnemonic, [Pure]> {
  let summary = summaryText;
  let arguments = (ins XT_RankedTensor:$input);
  let results = (outs XT_RankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "`(` $input `)` attr-dict `:` type($input) `->` type($result)";
}

def XT_ExpOp : XT_UnaryElementwiseOp<"exp", "elementwise tile exp">;
def XT_CosOp : XT_UnaryElementwiseOp<"cos", "elementwise tile cos">;
def XT_SinOp : XT_UnaryElementwiseOp<"sin", "elementwise tile sin">;
def XT_ReciprocalOp : XT_UnaryElementwiseOp<"reciprocal", "elementwise tile reciprocal">;
def XT_RsqrtOp : XT_UnaryElementwiseOp<"rsqrt", "elementwise tile rsqrt">;
def XT_SigmoidOp : XT_UnaryElementwiseOp<"sigmoid", "elementwise tile sigmoid">;
def XT_TanhOp : XT_UnaryElementwiseOp<"tanh", "elementwise tile tanh">;
def XT_SiluOp : XT_UnaryElementwiseOp<"silu", "elementwise tile silu">;
def XT_IToFOp : XT_UnaryCastOp<"itof", "convert integer tensor elements to floating point">;
def XT_FToIOp : XT_UnaryCastOp<"ftoi", "convert floating-point tensor elements to integer">;

def XT_MatmulOp : XT_Op<"matmul", [Pure]> {
  let summary = "tile matrix multiply";
  let arguments = (ins XT_RankedTensor:$lhs, XT_RankedTensor:$rhs);
  let results = (outs XT_RankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "`(` $lhs `,` $rhs `)` attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def XT_MMAOp : XT_Op<"mma", [Pure]> {
  let summary = "tile matrix multiply accumulate";
  let arguments = (ins XT_RankedTensor:$lhs, XT_RankedTensor:$rhs,
                       XT_RankedTensor:$acc);
  let results = (outs XT_RankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "`(` $lhs `,` $rhs `,` $acc `)` attr-dict `:` type($lhs) `,` type($rhs) `,` type($acc) `->` type($result)";
}

class XT_ReduceOp<string mnemonic, string summaryText> :
    XT_Op<mnemonic, [Pure]> {
  let summary = summaryText;
  let arguments = (ins XT_RankedTensor:$input, I64Attr:$axis);
  let results = (outs XT_RankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "`(` $input `)` attr-dict `:` type($input) `->` type($result)";
}

def XT_ReduceSumOp : XT_ReduceOp<"reduce_sum", "reduce a tensor axis by sum">;

def XT_ReduceMaxOp : XT_ReduceOp<"reduce_max", "reduce a tensor axis by max">;

def XT_ReshapeOp : XT_Op<"reshape", [Pure]> {
  let summary = "reshape a tensor tile";
  let arguments = (ins XT_RankedTensor:$input);
  let results = (outs XT_RankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "`(` $input `)` attr-dict `:` type($input) `->` type($result)";
}

def XT_TransposeOp : XT_Op<"transpose", [Pure]> {
  let summary = "transpose a rank-3 tensor tile by swapping the last two dimensions";
  let arguments = (ins XT_RankedTensor:$input);
  let results = (outs XT_RankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "`(` $input `)` attr-dict `:` type($input) `->` type($result)";
}

def XT_PermuteOp : XT_Op<"permute", [Pure]> {
  let summary = "permute tensor tile dimensions";
  let arguments = (ins XT_RankedTensor:$input,
                       DenseI64ArrayAttr:$permutation);
  let results = (outs XT_RankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "`(` $input `)` `{` `permutation` `=` $permutation `}` attr-dict `:` type($input) `->` type($result)";
}

#endif // XT_OPS
