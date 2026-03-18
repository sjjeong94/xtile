#include "nova/NovaDialect.h"
#include "nova/NovaPasses.h"
#include "xt/XTDialect.h"
#include "xt/XTPasses.h"

#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "mlir-c/Bindings/Python/Interop.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace {

nb::object getCapsuleFromModuleObject(nb::handle moduleObject) {
  if (PyCapsule_CheckExact(moduleObject.ptr()))
    return nb::borrow<nb::object>(moduleObject);

  if (!nb::hasattr(moduleObject, MLIR_PYTHON_CAPI_PTR_ATTR))
    throw nb::type_error("expected an mlir.ir.Module-compatible object");

  return moduleObject.attr(MLIR_PYTHON_CAPI_PTR_ATTR);
}

MlirModule unwrapModule(nb::handle moduleObject) {
  nb::object capsule = getCapsuleFromModuleObject(moduleObject);
  MlirModule module = mlirPythonCapsuleToModule(capsule.ptr());
  if (mlirModuleIsNull(module))
    throw nb::type_error("expected an mlir.ir.Module-compatible object");
  return module;
}

void loadRequiredDialects(mlir::MLIRContext &context) {
  context.getOrLoadDialect<mlir::nova::NovaDialect>();
  context.getOrLoadDialect<mlir::xt::XTDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();
}

std::string formatDiagnostic(mlir::Diagnostic &diagnostic) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  os << diagnostic.getLocation() << ": " << diagnostic;
  return buffer;
}

std::string moduleToString(MlirModule module) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  unwrap(module).print(os);
  os.flush();
  return buffer;
}

struct OwnedTestModule {
  std::unique_ptr<mlir::MLIRContext> context;
  MlirModule module;
};

nb::object createOwnedModuleCapsule(std::unique_ptr<OwnedTestModule> ownedModule) {
  MlirModule module = ownedModule->module;
  PyObject *capsule = PyCapsule_New(MLIR_PYTHON_GET_WRAPPED_POINTER(module),
                                    MLIR_PYTHON_CAPSULE_MODULE,
                                    [](PyObject *capsule) {
                                      void *context = PyCapsule_GetContext(capsule);
                                      if (!context)
                                        return;

                                      auto *ownedModule =
                                          static_cast<OwnedTestModule *>(context);
                                      if (!mlirModuleIsNull(ownedModule->module))
                                        mlirOperationDestroy(
                                            mlirModuleGetOperation(ownedModule->module));
                                      delete ownedModule;
                                    });
  if (!capsule)
    throw nb::python_error();
  if (PyCapsule_SetContext(capsule, ownedModule.release()) != 0) {
    Py_DECREF(capsule);
    throw nb::python_error();
  }
  return nb::steal<nb::object>(capsule);
}

nb::object parseModule(const std::string &moduleAsm) {
  auto context = std::make_unique<mlir::MLIRContext>();
  loadRequiredDialects(*context);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(moduleAsm, context.get());
  if (!module)
    throw nb::value_error("failed to parse test module");

  auto ownedModule = std::make_unique<OwnedTestModule>(OwnedTestModule{
      .context = std::move(context),
      .module = wrap(module.release()),
  });
  return createOwnedModuleCapsule(std::move(ownedModule));
}

nb::object runPass(nb::object moduleObject, const char *passName,
                   llvm::function_ref<void(mlir::PassManager &)> addPass) {
  MlirModule module = unwrapModule(moduleObject);
  mlir::ModuleOp moduleOp = unwrap(module);
  mlir::MLIRContext *context = moduleOp.getContext();
  loadRequiredDialects(*context);

  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(context, [&](mlir::Diagnostic &diag) {
    if (!diagnostics.empty())
      diagnostics.append("\n");
    diagnostics.append(formatDiagnostic(diag));
  });

  mlir::PassManager passManager(context);
  addPass(passManager);
  if (failed(passManager.run(moduleOp))) {
    if (diagnostics.empty())
      diagnostics = std::string(passName) + " failed";
    throw std::runtime_error(diagnostics);
  }

  return moduleObject;
}

nb::object toNova(nb::object moduleObject) {
  return runPass(moduleObject, "xtile.to_nova", [](mlir::PassManager &passManager) {
    passManager.nest<mlir::func::FuncOp>().addPass(
        mlir::xt::createXTToNovaPass());
  });
}

nb::object serialize(nb::object moduleObject) {
  return runPass(moduleObject, "xtile.serialize",
                 [](mlir::PassManager &passManager) {
                   passManager.nest<mlir::func::FuncOp>().addPass(
                       mlir::xt::createXTSerializePass());
                 });
}

nb::object novaOptimize(nb::object moduleObject) {
  return runPass(moduleObject, "xtile.nova_optimize",
                 [](mlir::PassManager &passManager) {
                   passManager.nest<mlir::func::FuncOp>().addPass(
                       mlir::nova::createNovaOptimizePass());
                 });
}

nb::object novaAllocate(nb::object moduleObject) {
  return runPass(moduleObject, "xtile.nova_allocate",
                 [](mlir::PassManager &passManager) {
                   passManager.nest<mlir::func::FuncOp>().addPass(
                       mlir::nova::createNovaAllocatePass());
                 });
}

} // namespace

NB_MODULE(_xtile, m) {
  m.def("parse", &parseModule, nb::arg("asm"),
        "Parse MLIR assembly into an xtile module object compatible with xtile passes.");
  m.def("to_nova", &toNova, nb::arg("module"),
        "Run the xt-to-nova lowering pass on an mlir.ir.Module.");
  m.def("serialize", &serialize, nb::arg("module"),
        "Run the xt-serialize pass on an mlir.ir.Module.");
  m.def("nova_optimize", &novaOptimize, nb::arg("module"),
        "Run the nova-optimize pass on an mlir.ir.Module.");
  m.def("nova_allocate", &novaAllocate, nb::arg("module"),
        "Run the nova-allocate pass on an mlir.ir.Module.");
  m.def("_parse_module", &parseModule, nb::arg("asm"));
  m.def("_module_asm",
        [](nb::object moduleObject) { return moduleToString(unwrapModule(moduleObject)); },
        nb::arg("module"));
}
