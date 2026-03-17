add_mlir_dialect_library(MLIRNova
  NovaAllocate.cpp
  NovaDialect.cpp
  NovaOptimize.cpp
  NovaOps.cpp
  NovaThreading.cpp

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
