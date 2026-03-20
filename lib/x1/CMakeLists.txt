add_mlir_dialect_library(MLIRX1
  X1Dialect.cpp
  X1Ops.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/x1

  DEPENDS
  MLIRX1OpsIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
)
