#include "xt/XTPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "xt/XTOps.h"

#include <map>
#include <optional>
#include <tuple>
#include <vector>

namespace mlir::xt {
#define GEN_PASS_DEF_XTSERIALIZE
#include "xt/XTPasses.h.inc"
} // namespace mlir::xt

using namespace mlir;

namespace {
class XTSerializePass
    : public mlir::xt::impl::XTSerializeBase<XTSerializePass> {
public:
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    auto grid = func->getAttrOfType<DenseI32ArrayAttr>("xt.grid");
    if (!grid) {
      func.emitOpError("xt-serialize requires func.func xt.grid attribute");
      signalPassFailure();
      return;
    }
    if (func.getNumResults() != 0) {
      func.emitOpError("xt-serialize only supports void functions");
      signalPassFailure();
      return;
    }
    if (!func.getBody().hasOneBlock()) {
      func.emitOpError("xt-serialize only supports single-block functions");
      signalPassFailure();
      return;
    }

    Block &entry = func.getBody().front();
    auto returnOp = dyn_cast<func::ReturnOp>(entry.getTerminator());
    if (!returnOp || returnOp.getNumOperands() != 0) {
      func.emitOpError("xt-serialize only supports void functions");
      signalPassFailure();
      return;
    }

    SmallVector<Operation *> originalOps;
    for (Operation &op : entry.without_terminator())
      originalOps.push_back(&op);

    OpBuilder builder(returnOp);
    auto makeI32Constant = [&](Location loc, int32_t value) {
      return builder.create<arith::ConstantIntOp>(loc, value, 32).getResult();
    };

    std::map<std::tuple<Operation *, int32_t, int32_t>, Value> sharedLoads;
    int32_t gridX = grid[0];
    int32_t gridY = grid[1];
    int32_t gridZ = grid[2];

    for (int32_t z = 0; z < gridZ; ++z) {
      for (int32_t y = 0; y < gridY; ++y) {
        for (int32_t x = 0; x < gridX; ++x) {
          IRMapping mapper;
          for (BlockArgument arg : entry.getArguments())
            mapper.map(arg, arg);

          Value xConst = makeI32Constant(func.getLoc(), x);
          Value yConst = makeI32Constant(func.getLoc(), y);
          Value zConst = makeI32Constant(func.getLoc(), z);

          for (Operation *op : originalOps) {
            if (auto blockIdOp = dyn_cast<xt::GetTileBlockIdOp>(op)) {
              mapper.map(blockIdOp.getResult(0), xConst);
              mapper.map(blockIdOp.getResult(1), yConst);
              mapper.map(blockIdOp.getResult(2), zConst);
              continue;
            }
            if (auto loadOp = dyn_cast<xt::LoadOp>(op)) {
              auto shared = loadOp.getSharedAttr();
              if (shared && shared.getInt() > 0) {
                int32_t canonicalY = shared.getInt() == 2 ? 0 : y;
                auto key = std::make_tuple(op, canonicalY, z);
                auto it = sharedLoads.find(key);
                if (it != sharedLoads.end()) {
                  mapper.map(loadOp.getResult(), it->second);
                  continue;
                }

                Operation *cloned = builder.clone(*op, mapper);
                Value result = cloned->getResult(0);
                sharedLoads.emplace(key, result);
                mapper.map(loadOp.getResult(), result);
                continue;
              }
            }
            Operation *cloned = builder.clone(*op, mapper);
            (void)cloned;
          }
        }
      }
    }

    for (Operation *op : llvm::reverse(originalOps))
      op->erase();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::xt::createXTSerializePass() {
  return std::make_unique<XTSerializePass>();
}
