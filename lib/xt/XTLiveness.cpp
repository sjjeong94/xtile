#include "xt/XTPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::xt {
#define GEN_PASS_DEF_XTLIVENESS
#include "xt/XTPasses.h.inc"
} // namespace mlir::xt

using namespace mlir;

namespace {
struct OpLivenessInfo {
  SmallVector<Value> liveIn;
  SmallVector<Value> liveOut;
};

static std::string formatValue(Value value, AsmState &asmState) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  value.printAsOperand(os, asmState);
  return buffer;
}

static std::string formatValueList(ArrayRef<Value> values, AsmState &asmState) {
  SmallVector<std::string> names;
  names.reserve(values.size());
  for (Value value : values)
    names.push_back(formatValue(value, asmState));
  llvm::sort(names);

  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  os << "[";
  for (size_t i = 0; i < names.size(); ++i) {
    if (i != 0)
      os << ", ";
    os << names[i];
  }
  os << "]";
  return buffer;
}

class XTLivenessPass
    : public mlir::xt::impl::XTLivenessBase<XTLivenessPass> {
public:
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    AsmState asmState(func);
    llvm::outs() << "liveness @" << func.getSymName() << "\n";

    for (Block &block : func.getBlocks()) {
      llvm::DenseMap<Operation *, OpLivenessInfo> opInfo;
      SmallVector<Value> live;
      llvm::SmallPtrSet<Value, 16> liveSet;

      auto addLive = [&](Value value) {
        if (liveSet.insert(value).second)
          live.push_back(value);
      };
      auto removeDefined = [&](Value value) {
        if (!liveSet.erase(value))
          return;
        llvm::erase(live, value);
      };

      for (Operation &op : llvm::reverse(block)) {
        OpLivenessInfo info;
        info.liveOut = live;

        for (Value result : op.getResults())
          removeDefined(result);
        for (Value operand : op.getOperands())
          addLive(operand);

        info.liveIn = live;
        opInfo[&op] = std::move(info);
      }

      for (Operation &op : block) {
        const OpLivenessInfo &info = opInfo[&op];
        llvm::outs() << "  op: " << op.getName().getStringRef()
                     << " live_in=" << formatValueList(info.liveIn, asmState)
                     << " live_out=" << formatValueList(info.liveOut, asmState)
                     << "\n";
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::xt::createXTLivenessPass() {
  return std::make_unique<XTLivenessPass>();
}
