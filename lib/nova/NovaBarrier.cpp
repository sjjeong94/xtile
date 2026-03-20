#include "nova/NovaPasses.h"
#include "nova/NovaOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

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

static bool isSkippedComputeOp(Operation *op) {
  return isa<nova::LoadOp, nova::StoreOp, nova::FreeOp, nova::BarrierOp>(op);
}

class NovaBarrierPass : public mlir::nova::impl::NovaBarrierBase<NovaBarrierPass> {
public:
  void runOnOperation() override {
    func::FuncOp func = getOperation();

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

        Operation *next = op->getNextNode();
        if (isBarrierWithMode(next, 0))
          continue;

        builder.setInsertionPointAfter(op);
        builder.create<nova::BarrierOp>(
            op->getLoc(), builder.getI32IntegerAttr(0));
      }

      for (Operation &op : llvm::make_early_inc_range(block)) {
        auto ret = dyn_cast<func::ReturnOp>(&op);
        if (!ret)
          continue;

        Operation *prev = ret->getPrevNode();
        if (isBarrierWithMode(prev, 1))
          continue;

        builder.setInsertionPoint(ret);
        builder.create<nova::BarrierOp>(
            ret.getLoc(), builder.getI32IntegerAttr(1));
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::nova::createNovaBarrierPass() {
  return std::make_unique<NovaBarrierPass>();
}
