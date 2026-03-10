add_mlir_dialect(XTOps xt)
add_mlir_doc(XTDialect XTDialect xt/ -gen-dialect-doc)
add_mlir_doc(XTOps XTOps xt/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS XTPasses.td)
mlir_tablegen(XTPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRXTPassesIncGen)
