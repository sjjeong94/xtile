add_mlir_dialect_library(MLIRXT
  XTDialect.cpp
  XTOps.cpp
  XTLowerToLoops.cpp
  XTToNova.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/xt

  DEPENDS
  MLIRXTOpsIncGen
  MLIRXTPassesIncGen

  LINK_LIBS PUBLIC
  MLIRNova
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRRewrite
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransforms
)
