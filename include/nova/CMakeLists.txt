add_mlir_dialect(NovaOps nova)
add_mlir_doc(NovaDialect NovaDialect nova/ -gen-dialect-doc)
add_mlir_doc(NovaOps NovaOps nova/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS NovaPasses.td)
mlir_tablegen(NovaPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRNovaPassesIncGen)
