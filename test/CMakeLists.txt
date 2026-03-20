llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
)

enable_testing()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

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

if(TARGET xt)
  list(APPEND XT_TEST_DEPENDS xt)
endif()

add_lit_testsuite(check-xt "Running xt regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${XT_TEST_DEPENDS}
)
set_target_properties(check-xt PROPERTIES FOLDER "Tests")

add_lit_testsuites(XT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${XT_TEST_DEPENDS})

if(TARGET xt)
  add_test(
    NAME xt-python-to-nova
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/to_nova.py
  )
  add_test(
    NAME xt-python-serialize
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/serialize.py
  )
  add_test(
    NAME xt-python-optimize
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/optimize.py
  )
  add_test(
    NAME xt-python-allocate
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/allocate.py
  )
  add_test(
    NAME xt-python-threading
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/threading.py
  )
  add_test(
    NAME xt-python-barrier
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/barrier.py
  )
  add_test(
    NAME xt-python-to-x1
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/to_x1.py
  )
  add_test(
    NAME xt-python-compile
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/compile.py
  )
  add_test(
    NAME xt-python-dsl-convert
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/dsl_convert.py
  )
  set_tests_properties(xt-python-to-nova PROPERTIES
    ENVIRONMENT "PYTHONPATH=${XT_BINARY_DIR}/python"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(xt-python-serialize PROPERTIES
    ENVIRONMENT "PYTHONPATH=${XT_BINARY_DIR}/python"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(xt-python-optimize PROPERTIES
    ENVIRONMENT "PYTHONPATH=${XT_BINARY_DIR}/python"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(xt-python-allocate PROPERTIES
    ENVIRONMENT "PYTHONPATH=${XT_BINARY_DIR}/python"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(xt-python-threading PROPERTIES
    ENVIRONMENT "PYTHONPATH=${XT_BINARY_DIR}/python"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(xt-python-barrier PROPERTIES
    ENVIRONMENT "PYTHONPATH=${XT_BINARY_DIR}/python"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(xt-python-to-x1 PROPERTIES
    ENVIRONMENT "PYTHONPATH=${XT_BINARY_DIR}/python"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(xt-python-compile PROPERTIES
    ENVIRONMENT "PYTHONPATH=${XT_BINARY_DIR}/python"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  set_tests_properties(xt-python-dsl-convert PROPERTIES
    ENVIRONMENT "PYTHONPATH=${XT_BINARY_DIR}/python"
    WORKING_DIRECTORY "${XT_SOURCE_DIR}"
  )
endif()
