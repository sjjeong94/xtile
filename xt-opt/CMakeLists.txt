set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_executable(xt-opt xt-opt.cpp)
llvm_update_compile_flags(xt-opt)
target_link_libraries(xt-opt PRIVATE
  MLIRNova
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRSupport
  MLIRTransforms
  MLIRXT
  MLIROptLib
)

mlir_check_all_link_libraries(xt-opt)
