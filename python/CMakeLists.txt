set(XT_PYTHON_OUTPUT_DIR "${XT_BINARY_DIR}/python")
set(XT_PYTHON_PACKAGE_DIR "${XT_PYTHON_OUTPUT_DIR}/xtile")
set(XT_PYTHON_PACKAGE_SOURCES
  "${XT_SOURCE_DIR}/xtile/__init__.py"
  "${XT_SOURCE_DIR}/xtile/dsl.py"
)

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
  LIBRARY_OUTPUT_DIRECTORY "${XT_PYTHON_PACKAGE_DIR}"
  OUTPUT_NAME "_xtile"
  NO_SONAME ON
)

add_custom_command(
  OUTPUT
    "${XT_PYTHON_PACKAGE_DIR}/__init__.py"
    "${XT_PYTHON_PACKAGE_DIR}/dsl.py"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${XT_PYTHON_PACKAGE_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${XT_SOURCE_DIR}/xtile/__init__.py"
    "${XT_PYTHON_PACKAGE_DIR}/__init__.py"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${XT_SOURCE_DIR}/xtile/dsl.py"
    "${XT_PYTHON_PACKAGE_DIR}/dsl.py"
  DEPENDS ${XT_PYTHON_PACKAGE_SOURCES}
)

add_custom_target(xt_python_package ALL
  DEPENDS
    "${XT_PYTHON_PACKAGE_DIR}/__init__.py"
    "${XT_PYTHON_PACKAGE_DIR}/dsl.py"
)

add_dependencies(xt xt_python_package)
