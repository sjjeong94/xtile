add_mlir_dialect_library(MLIRNova
  NovaDialect.cpp
  NovaOptimize.cpp
  NovaSerialize.cpp
  NovaOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/nova

  DEPENDS
  MLIRNovaOpsIncGen
  MLIRNovaPassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
)
