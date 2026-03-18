cmake_minimum_required(VERSION 3.20.0)
project(xt LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

set(LLVM_LIT_ARGS "-sv -j 1" CACHE STRING "Default options for lit")

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include(MLIRDetectPythonEnv)

set(XT_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(XT_BINARY_DIR ${PROJECT_BINARY_DIR})
get_filename_component(XT_LLVM_CMAKE_DIR_PARENT "${LLVM_DIR}" DIRECTORY)
get_filename_component(XT_LLVM_CMAKE_ROOT "${XT_LLVM_CMAKE_DIR_PARENT}/../.." ABSOLUTE)

option(XT_ENABLE_PYTHON_BINDINGS "Build xt nanobind Python bindings" ON)

if(XT_ENABLE_PYTHON_BINDINGS)
  find_package(Python3 ${LLVM_MINIMUM_PYTHON_VERSION}
    COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(Python ${LLVM_MINIMUM_PYTHON_VERSION}
    COMPONENTS Interpreter Development.Module REQUIRED)
  mlir_detect_nanobind_install()
  find_package(nanobind 2.4 CONFIG REQUIRED)
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${XT_SOURCE_DIR}/include)
include_directories(${XT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(xt-opt)
if(XT_ENABLE_PYTHON_BINDINGS)
  add_subdirectory(python)
endif()
add_subdirectory(test)
