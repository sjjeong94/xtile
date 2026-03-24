add_mlir_dialect_library(MLIRNova
  NovaAllocate.cpp
  NovaBarrier.cpp
  NovaDialect.cpp
  NovaOptimize.cpp
  NovaOps.cpp
  NovaThreading.cpp
  NovaToX1.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/nova

  DEPENDS
  MLIRNovaAttrDefsIncGen
  MLIRNovaOpsIncGen
  MLIRNovaPassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRTransforms
  MLIRX1
)
