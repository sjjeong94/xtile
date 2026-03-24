#ifndef NOVA_DIALECT
#define NOVA_DIALECT

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

def NovaDialect : Dialect {
  let name = "nova";
  let cppNamespace = "::mlir::nova";
  let useDefaultAttributePrinterParser = 1;
  let summary = "Nova hardware-oriented tile dialect";
  let description = [{
    The nova dialect models hardware-oriented tile operations produced from xt.
  }];
}

class Nova_Attr<string name, string attrMnemonic, list<Trait> traits = []> :
    AttrDef<NovaDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Nova_TensorLayoutAttr : Nova_Attr<"TensorLayout", "tensor_layout"> {
  let summary = "Tensor encoding metadata for Nova tensors";
  let description = [{
    Captures optional bank and slice metadata and the target memory space
    for a Nova ranked tensor encoding.
  }];

  let parameters = (ins
    OptionalParameter<"::mlir::IntegerAttr">:$bank0,
    OptionalParameter<"::mlir::IntegerAttr">:$bank1,
    OptionalParameter<"::mlir::DenseI64ArrayAttr">:$start0,
    OptionalParameter<"::mlir::DenseI64ArrayAttr">:$start1,
    OptionalParameter<"::mlir::DenseI64ArrayAttr">:$shape0,
    OptionalParameter<"::mlir::DenseI64ArrayAttr">:$shape1,
    OptionalParameter<"::mlir::IntegerAttr">:$space
  );
  let hasCustomAssemblyFormat = 1;
}

#endif // NOVA_DIALECT
