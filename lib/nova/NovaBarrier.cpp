#include "nova/NovaPasses.h"
#include "nova/NovaOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::nova {
#define GEN_PASS_DEF_NOVABARRIER
#include "nova/NovaPasses.h.inc"
} // namespace mlir::nova

using namespace mlir;

namespace {
static bool isBarrierWithMode(Operation *op, int32_t mode) {
  auto barrier = dyn_cast_or_null<nova::BarrierOp>(op);
  return barrier && static_cast<int32_t>(barrier.getMode()) == mode;
}

static Operation *getNextNonKeepAlive(Operation *op) {
  for (Operation *next = op ? op->getNextNode() : nullptr; next;
       next = next->getNextNode()) {
    if (!isa<nova::KeepAliveOp>(next))
      return next;
  }
  return nullptr;
}

static Operation *getPrevNonKeepAlive(Operation *op) {
  for (Operation *prev = op ? op->getPrevNode() : nullptr; prev;
       prev = prev->getPrevNode()) {
    if (!isa<nova::KeepAliveOp>(prev))
      return prev;
  }
  return nullptr;
}

static bool isSkippedComputeOp(Operation *op) {
  return isa<nova::LoadOp, nova::StoreOp, nova::BarrierOp, nova::KeepAliveOp>(op);
}

static void collectTensorValues(Operation *op, llvm::SetVector<Value> &tensorSet) {
  auto appendIfTensor = [&](Value value) {
    if (isa<RankedTensorType>(value.getType()))
      tensorSet.insert(value);
  };

  for (Value operand : op->getOperands())
    appendIfTensor(operand);
  for (Value result : op->getResults())
    appendIfTensor(result);
}

static void insertKeepAliveAfterBarrier(nova::BarrierOp barrier,
                                        const llvm::SetVector<Value> &tensorKeep) {
  if (tensorKeep.empty())
    return;
  if (isa_and_nonnull<nova::KeepAliveOp>(barrier->getPrevNode()))
    return;

  SmallVector<Value> operands(tensorKeep.begin(), tensorKeep.end());
  OpBuilder builder(barrier);
  builder.setInsertionPoint(barrier);
  builder.create<nova::KeepAliveOp>(barrier.getLoc(), TypeRange{}, operands);
}

static bool isDoubleBufferingDisabled(func::FuncOp func) {
  auto attr = func->getAttrOfType<IntegerAttr>("xt.double_buffering");
  return !attr || attr.getInt() == 0;
}

class NovaBarrierPass : public mlir::nova::impl::NovaBarrierBase<NovaBarrierPass> {
public:
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    bool doubleBufferingDisabled = isDoubleBufferingDisabled(func);

    for (Block &block : func.getBody()) {
      SmallVector<Operation *> ops;
      ops.reserve(block.getOperations().size());
      for (Operation &op : block)
        ops.push_back(&op);

      OpBuilder builder(func.getContext());

      for (Operation *op : ops) {
        if (isa<func::ReturnOp>(op))
          continue;
        if (!op->getDialect() || op->getDialect()->getNamespace() != StringRef("nova"))
          continue;
        if (isSkippedComputeOp(op))
          continue;

        Operation *next = getNextNonKeepAlive(op);
        if (isBarrierWithMode(next, 0))
          continue;

        builder.setInsertionPointAfter(op);
        builder.create<nova::BarrierOp>(
            op->getLoc(), builder.getI32IntegerAttr(0));
      }

      if (doubleBufferingDisabled) {
        for (Operation *op : ops) {
          auto store = dyn_cast<nova::StoreOp>(op);
          if (!store)
            continue;

          Operation *next = getNextNonKeepAlive(op);
          if (isBarrierWithMode(next, 1))
            continue;

          builder.setInsertionPointAfter(op);
          builder.create<nova::BarrierOp>(
              store.getLoc(), builder.getI32IntegerAttr(1));
        }
      }

      for (Operation &op : llvm::make_early_inc_range(block)) {
        auto ret = dyn_cast<func::ReturnOp>(&op);
        if (!ret)
          continue;

        Operation *prev = getPrevNonKeepAlive(ret);
        if (isBarrierWithMode(prev, 1))
          continue;

        builder.setInsertionPoint(ret);
        builder.create<nova::BarrierOp>(
            ret.getLoc(), builder.getI32IntegerAttr(1));
      }

      if (doubleBufferingDisabled)
        continue;

      llvm::SetVector<Value> tensorKeep;
      llvm::SetVector<Value> tensorSet;
      for (Operation &op : llvm::make_early_inc_range(block)) {
        if (!isa<nova::LoadOp, nova::BarrierOp, nova::KeepAliveOp>(&op))
          collectTensorValues(&op, tensorSet);

        auto barrier = dyn_cast<nova::BarrierOp>(&op);
        if (!barrier)
          continue;

        insertKeepAliveAfterBarrier(barrier, tensorKeep);
        tensorKeep.clear();
        if (static_cast<int32_t>(barrier.getMode()) == 0) {
          tensorKeep = tensorSet;
          tensorSet.clear();
          continue;
        }

        tensorKeep.clear();
        tensorSet.clear();
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::nova::createNovaBarrierPass() {
  return std::make_unique<NovaBarrierPass>();
}
