#include "xt/XTDialect.h"
#include "xt/XTPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::xt::registerPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::xt::XTDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "xt optimizer driver\n", registry));
}
