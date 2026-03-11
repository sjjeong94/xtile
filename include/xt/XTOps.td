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
  let hasCustomAssemblyFormat = 1;
}

def XT_LoadOp : XT_Op<"load"> {
  let summary = "load a tensor tile from a memref";
  let arguments = (ins XT_RankedMemRef:$source,
                       Variadic<XT_AnyCoord>:$coords,
                       OptionalAttr<I64Attr>:$shared);
  let results = (outs XT_RankedTensor:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def XT_StoreOp : XT_Op<"store"> {
  let summary = "store a tensor tile into a memref";
  let arguments = (ins XT_RankedTensor:$value,
                       XT_RankedMemRef:$dest,
                       Variadic<XT_AnyCoord>:$coords);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def XT_AddOp : XT_Op<"add", [Pure]> {
  let summary = "elementwise tile add";
  let arguments = (ins XT_RankedTensor:$lhs, XT_RankedTensor:$rhs);
  let results = (outs XT_RankedTensor:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def XT_SubOp : XT_Op<"sub", [Pure]> {
  let summary = "elementwise tile sub";
  let arguments = (ins XT_RankedTensor:$lhs, XT_RankedTensor:$rhs);
  let results = (outs XT_RankedTensor:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def XT_MulOp : XT_Op<"mul", [Pure]> {
  let summary = "elementwise tile mul";
  let arguments = (ins XT_RankedTensor:$lhs, XT_RankedTensor:$rhs);
  let results = (outs XT_RankedTensor:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def XT_ExpOp : XT_Op<"exp", [Pure]> {
  let summary = "elementwise tile exp";
  let arguments = (ins XT_RankedTensor:$input);
  let results = (outs XT_RankedTensor:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

class XT_UnaryElementwiseOp<string mnemonic, string summaryText> :
    XT_Op<mnemonic, [Pure]> {
  let summary = summaryText;
  let arguments = (ins XT_RankedTensor:$input);
  let results = (outs XT_RankedTensor:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def XT_CosOp : XT_UnaryElementwiseOp<"cos", "elementwise tile cos">;
def XT_SinOp : XT_UnaryElementwiseOp<"sin", "elementwise tile sin">;
def XT_ReciprocalOp : XT_UnaryElementwiseOp<"reciprocal", "elementwise tile reciprocal">;
def XT_RsqrtOp : XT_UnaryElementwiseOp<"rsqrt", "elementwise tile rsqrt">;
def XT_SigmoidOp : XT_UnaryElementwiseOp<"sigmoid", "elementwise tile sigmoid">;
def XT_TanhOp : XT_UnaryElementwiseOp<"tanh", "elementwise tile tanh">;
def XT_SiluOp : XT_UnaryElementwiseOp<"silu", "elementwise tile silu">;

def XT_MatmulOp : XT_Op<"matmul", [Pure]> {
  let summary = "tile matrix multiply";
  let arguments = (ins XT_RankedTensor:$lhs, XT_RankedTensor:$rhs);
  let results = (outs XT_RankedTensor:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def XT_MMAOp : XT_Op<"mma", [Pure]> {
  let summary = "tile matrix multiply accumulate";
  let arguments = (ins XT_RankedTensor:$lhs, XT_RankedTensor:$rhs,
                       XT_RankedTensor:$acc);
  let results = (outs XT_RankedTensor:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def XT_Conv2DOp : XT_Op<"conv2d", [Pure]> {
  let summary = "tile 2d convolution";
  let arguments = (ins XT_RankedTensor:$input, XT_RankedTensor:$filter,
                       DenseI64ArrayAttr:$pad,
                       DenseI64ArrayAttr:$stride,
                       DenseI64ArrayAttr:$dilation);
  let results = (outs XT_RankedTensor:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def XT_DepthwiseConv2DOp : XT_Op<"depthwise_conv2d", [Pure]> {
  let summary = "tile depthwise 2d convolution";
  let arguments = (ins XT_RankedTensor:$input, XT_RankedTensor:$filter,
                       DenseI64ArrayAttr:$pad,
                       DenseI64ArrayAttr:$stride,
                       DenseI64ArrayAttr:$dilation);
  let results = (outs XT_RankedTensor:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def XT_ReduceSumOp : XT_Op<"reduce_sum", [Pure]> {
  let summary = "reduce the last dimension by sum";
  let arguments = (ins XT_RankedTensor:$input);
  let results = (outs XT_RankedTensor:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def XT_ReduceMaxOp : XT_Op<"reduce_max", [Pure]> {
  let summary = "reduce the last dimension by max";
  let arguments = (ins XT_RankedTensor:$input);
  let results = (outs XT_RankedTensor:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // XT_OPS
