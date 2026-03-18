set(XT_PYTHON_OUTPUT_DIR "${XT_BINARY_DIR}/python")

nanobind_add_module(xt
  FREE_THREADED
  XTModule.cpp
)

target_compile_options(xt PRIVATE
  -frtti
  -fexceptions
)

target_include_directories(xt PRIVATE
  ${XT_SOURCE_DIR}/include
  ${XT_BINARY_DIR}/include
)

target_link_libraries(xt PRIVATE
  LLVMSupport
  MLIRArithDialect
  MLIRCAPIIR
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransformUtils
  MLIRTransforms
  MLIRXT
)

set_target_properties(xt PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${XT_PYTHON_OUTPUT_DIR}"
  OUTPUT_NAME "xtile"
  NO_SONAME ON
)
