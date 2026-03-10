llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(XT_TEST_DEPENDS
  FileCheck
  count
  not
  xt-opt
)

add_lit_testsuite(check-xt "Running xt regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${XT_TEST_DEPENDS}
)
set_target_properties(check-xt PROPERTIES FOLDER "Tests")

add_lit_testsuites(XT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${XT_TEST_DEPENDS})
