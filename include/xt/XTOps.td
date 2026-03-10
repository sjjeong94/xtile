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
                       XT_AnyCoord:$tileX,
                       XT_AnyCoord:$tileY,
                       DenseI64ArrayAttr:$tile);
  let results = (outs XT_RankedTensor:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def XT_StoreOp : XT_Op<"store"> {
  let summary = "store a tensor tile into a memref";
  let arguments = (ins XT_RankedTensor:$value,
                       XT_RankedMemRef:$dest,
                       XT_AnyCoord:$tileX,
                       XT_AnyCoord:$tileY,
                       DenseI64ArrayAttr:$tile);
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

def XT_ExpOp : XT_Op<"exp", [Pure]> {
  let summary = "elementwise tile exp";
  let arguments = (ins XT_RankedTensor:$input);
  let results = (outs XT_RankedTensor:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // XT_OPS
