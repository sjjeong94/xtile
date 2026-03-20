#include "nova/NovaPasses.h"
#include "nova/NovaOps.h"
#include "x1/X1Ops.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::nova {
#define GEN_PASS_DEF_NOVATOX1
#include "nova/NovaPasses.h.inc"
} // namespace mlir::nova

using namespace mlir;

namespace {
struct TensorLayout {
  int64_t baseBank = 0;
  int64_t space = 0;
  int64_t threadCount = 1;
  int64_t chunkRows = 0;
  SmallVector<int64_t> shape;
};

struct MemRefLayout {
  TensorLayout tensor;
  SmallVector<int64_t> start;
};

static std::optional<int64_t> getEncodingI64(Type type, StringRef name) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType)
    return std::nullopt;

  auto encoding = dyn_cast_or_null<DictionaryAttr>(tensorType.getEncoding());
  if (!encoding)
    return std::nullopt;

  auto attr = dyn_cast_or_null<IntegerAttr>(encoding.get(name));
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

static FailureOr<SmallVector<int64_t>>
toIntVector(Operation *op, DenseI64ArrayAttr attr, StringRef name) {
  if (!attr) {
    op->emitOpError() << "requires " << name << " attribute";
    return failure();
  }

  return SmallVector<int64_t>(attr.asArrayRef().begin(), attr.asArrayRef().end());
}

static FailureOr<TensorLayout> getTensorLayoutFromType(Operation *op, Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || !tensorType.hasStaticShape() || tensorType.getRank() < 1) {
    op->emitOpError("requires statically shaped ranked tensor type");
    return failure();
  }

  TensorLayout layout;
  layout.baseBank = getEncodingI64(type, "bank").value_or(0);
  layout.space = getEncodingI64(type, "space").value_or(0);
  layout.shape.assign(tensorType.getShape().begin(), tensorType.getShape().end());

  int64_t fullRows = layout.shape.front();
  int64_t threading = getEncodingI64(type, "threading").value_or(fullRows);
  if (threading <= 0) {
    op->emitOpError("requires positive threading when present");
    return failure();
  }

  layout.chunkRows = std::min(fullRows, threading);
  layout.threadCount = std::max<int64_t>(1, llvm::divideCeil(fullRows, threading));
  return layout;
}

static FailureOr<MemRefLayout> getMemRefLayout(Operation *op, Type type,
                                               DenseI64ArrayAttr startAttr) {
  FailureOr<TensorLayout> tensor = getTensorLayoutFromType(op, type);
  if (failed(tensor))
    return failure();

  FailureOr<SmallVector<int64_t>> start = toIntVector(op, startAttr, "start");
  if (failed(start))
    return failure();
  if (start->size() != tensor->shape.size()) {
    op->emitOpError("requires start rank to match tensor rank");
    return failure();
  }

  return MemRefLayout{*tensor, std::move(*start)};
}

static FailureOr<TensorLayout> propagateLayout(Operation *op, Type resultType,
                                               const TensorLayout &inputLayout) {
  FailureOr<TensorLayout> result = getTensorLayoutFromType(op, resultType);
  if (failed(result))
    return failure();

  result->threadCount = inputLayout.threadCount;
  if (!result->shape.empty())
    result->chunkRows =
        inputLayout.threadCount > 1 ? inputLayout.chunkRows : result->shape.front();
  return result;
}

static FailureOr<TensorLayout> propagateBinaryLayout(Operation *op, Type resultType,
                                                     const TensorLayout &lhs,
                                                     const TensorLayout &rhs) {
  if (lhs.threadCount != rhs.threadCount) {
    op->emitOpError("requires matching x1 lane counts");
    return failure();
  }
  return propagateLayout(op, resultType, lhs);
}

static int64_t getFirstBank(const TensorLayout &layout) {
  return layout.baseBank;
}

static int64_t getLastBank(const TensorLayout &layout) {
  return layout.baseBank + layout.threadCount - 1;
}

static int64_t getMatrixRowsPerLane(const TensorLayout &layout) {
  return layout.threadCount > 1 ? layout.chunkRows : layout.shape.front();
}

static ArrayAttr getI64ArrayAttr(OpBuilder &builder, ArrayRef<int64_t> values) {
  SmallVector<Attribute> attrs;
  attrs.reserve(values.size());
  for (int64_t value : values)
    attrs.push_back(builder.getI64IntegerAttr(value));
  return builder.getArrayAttr(attrs);
}

static SmallVector<int64_t> getChunkStart(const MemRefLayout &layout,
                                          int64_t thread) {
  SmallVector<int64_t> start = layout.start;
  if (!start.empty())
    start.front() += thread * layout.tensor.chunkRows;
  return start;
}

static SmallVector<int64_t> getChunkShape(const TensorLayout &layout,
                                          int64_t thread) {
  SmallVector<int64_t> shape = layout.shape;
  if (!shape.empty()) {
    int64_t consumed = thread * layout.chunkRows;
    shape.front() = std::min(layout.chunkRows, layout.shape.front() - consumed);
  }
  return shape;
}

static void createX1Load(OpBuilder &builder, Location loc, Value source,
                         const MemRefLayout &layout) {
  for (int64_t thread = 0; thread < layout.tensor.threadCount; ++thread) {
    builder.create<x1::LoadOp>(
        loc, source, builder.getI64IntegerAttr(layout.tensor.baseBank + thread),
        builder.getI64IntegerAttr(layout.tensor.space),
        builder.getI64IntegerAttr(thread),
        getI64ArrayAttr(builder, getChunkStart(layout, thread)),
        getI64ArrayAttr(builder, getChunkShape(layout.tensor, thread)));
  }
}

static void createX1Store(OpBuilder &builder, Location loc, Value dest,
                          const MemRefLayout &layout) {
  for (int64_t thread = 0; thread < layout.tensor.threadCount; ++thread) {
    builder.create<x1::StoreOp>(
        loc, dest, builder.getI64IntegerAttr(layout.tensor.baseBank + thread),
        builder.getI64IntegerAttr(layout.tensor.space),
        builder.getI64IntegerAttr(thread),
        getI64ArrayAttr(builder, getChunkStart(layout, thread)),
        getI64ArrayAttr(builder, getChunkShape(layout.tensor, thread)));
  }
}

static void createX1Reduce(OpBuilder &builder, nova::ReduceOp op,
                           const TensorLayout &inputLayout,
                           const TensorLayout &resultLayout) {
  builder.create<x1::ReduceOp>(
      op.getLoc(), builder.getI64IntegerAttr(getFirstBank(inputLayout)),
      builder.getI64IntegerAttr(getLastBank(inputLayout)),
      builder.getI64IntegerAttr(getFirstBank(resultLayout)),
      builder.getI64IntegerAttr(getLastBank(resultLayout)),
      builder.getI64IntegerAttr(getMatrixRowsPerLane(inputLayout)),
      builder.getI64IntegerAttr(inputLayout.shape[1]), op.getModeAttr());
}

static void createX1Broadcast(OpBuilder &builder, nova::BroadcastOp op,
                              const TensorLayout &lhsLayout,
                              const TensorLayout &rhsLayout,
                              const TensorLayout &resultLayout) {
  builder.create<x1::BroadcastOp>(
      op.getLoc(), builder.getI64IntegerAttr(getFirstBank(lhsLayout)),
      builder.getI64IntegerAttr(getLastBank(lhsLayout)),
      builder.getI64IntegerAttr(getFirstBank(rhsLayout)),
      builder.getI64IntegerAttr(getLastBank(rhsLayout)),
      builder.getI64IntegerAttr(getFirstBank(resultLayout)),
      builder.getI64IntegerAttr(getLastBank(resultLayout)), op.getLhsAAttr(),
      op.getLhsBAttr(), op.getModeAttr(), op.getRhsAAttr(), op.getRhsBAttr());
}

template <typename X1Op, typename NovaOp>
static void createX1UnaryBankCommand(OpBuilder &builder, NovaOp op,
                                     const TensorLayout &inputLayout,
                                     const TensorLayout &resultLayout,
                                     int64_t m, int64_t n) {
  builder.create<X1Op>(
      op.getLoc(), builder.getI64IntegerAttr(getFirstBank(inputLayout)),
      builder.getI64IntegerAttr(getLastBank(inputLayout)),
      builder.getI64IntegerAttr(getFirstBank(resultLayout)),
      builder.getI64IntegerAttr(getLastBank(resultLayout)),
      builder.getI64IntegerAttr(m), builder.getI64IntegerAttr(n));
}

static LogicalResult createX1Matmul(OpBuilder &builder, nova::MatmulOp op,
                                    const TensorLayout &lhsLayout,
                                    const TensorLayout &rhsLayout,
                                    const TensorLayout &resultLayout) {
  auto lhsType = dyn_cast<RankedTensorType>(op.getLhs().getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!lhsType || !lhsType.hasStaticShape() || lhsType.getRank() < 2 ||
      !resultType || !resultType.hasStaticShape() || resultType.getRank() < 2) {
    op.emitOpError("requires statically shaped rank-2 matmul operands/results");
    return failure();
  }

  builder.create<x1::MatmulOp>(
      op.getLoc(), builder.getI64IntegerAttr(getFirstBank(lhsLayout)),
      builder.getI64IntegerAttr(getLastBank(lhsLayout)),
      builder.getI64IntegerAttr(getFirstBank(rhsLayout)),
      builder.getI64IntegerAttr(getLastBank(rhsLayout)),
      builder.getI64IntegerAttr(getFirstBank(resultLayout)),
      builder.getI64IntegerAttr(getLastBank(resultLayout)),
      builder.getI64IntegerAttr(getMatrixRowsPerLane(lhsLayout)),
      builder.getI64IntegerAttr(resultType.getShape()[1]),
      builder.getI64IntegerAttr(lhsType.getShape()[1]));
  return success();
}

class NovaToX1Pass : public mlir::nova::impl::NovaToX1Base<NovaToX1Pass> {
public:
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (!func.getBody().hasOneBlock()) {
      func.emitOpError("nova-to-x1 only supports single-block functions");
      signalPassFailure();
      return;
    }

    Block &block = func.getBody().front();
    SmallVector<Operation *> ops;
    ops.reserve(block.getOperations().size());
    for (Operation &op : block)
      ops.push_back(&op);

    DenseMap<Value, TensorLayout> layouts;
    SmallVector<Operation *> eraseOps;
    OpBuilder builder(func.getContext());

    for (Operation *operation : ops) {
      builder.setInsertionPoint(operation);

      if (auto load = dyn_cast<nova::LoadOp>(operation)) {
        FailureOr<MemRefLayout> layout =
            getMemRefLayout(load, load.getResult().getType(), load.getStartAttr());
        if (failed(layout)) {
          signalPassFailure();
          return;
        }
        createX1Load(builder, load.getLoc(), load.getSource(), *layout);
        layouts[load.getResult()] = layout->tensor;
        eraseOps.push_back(operation);
        continue;
      }

      if (auto reduce = dyn_cast<nova::ReduceOp>(operation)) {
        auto inputIt = layouts.find(reduce.getInput());
        if (inputIt == layouts.end()) {
          reduce.emitOpError("requires lowered input metadata");
          signalPassFailure();
          return;
        }

        FailureOr<TensorLayout> resultLayout =
            propagateLayout(reduce, reduce.getResult().getType(), inputIt->second);
        if (failed(resultLayout)) {
          signalPassFailure();
          return;
        }
        createX1Reduce(builder, reduce, inputIt->second, *resultLayout);
        layouts[reduce.getResult()] = *resultLayout;
        eraseOps.push_back(operation);
        continue;
      }

      if (auto broadcast = dyn_cast<nova::BroadcastOp>(operation)) {
        auto lhsIt = layouts.find(broadcast.getLhs());
        auto rhsIt = layouts.find(broadcast.getRhs());
        if (lhsIt == layouts.end() || rhsIt == layouts.end()) {
          broadcast.emitOpError("requires lowered input metadata");
          signalPassFailure();
          return;
        }

        FailureOr<TensorLayout> resultLayout = propagateBinaryLayout(
            broadcast, broadcast.getResult().getType(), lhsIt->second, rhsIt->second);
        if (failed(resultLayout)) {
          signalPassFailure();
          return;
        }
        createX1Broadcast(builder, broadcast, lhsIt->second, rhsIt->second,
                          *resultLayout);
        layouts[broadcast.getResult()] = *resultLayout;
        eraseOps.push_back(operation);
        continue;
      }

      if (auto exp = dyn_cast<nova::ExpOp>(operation)) {
        auto inputIt = layouts.find(exp.getInput());
        if (inputIt == layouts.end()) {
          exp.emitOpError("requires lowered input metadata");
          signalPassFailure();
          return;
        }

        FailureOr<TensorLayout> resultLayout =
            propagateLayout(exp, exp.getResult().getType(), inputIt->second);
        if (failed(resultLayout)) {
          signalPassFailure();
          return;
        }
        createX1UnaryBankCommand<x1::ExpOp>(
            builder, exp, inputIt->second, *resultLayout,
            getMatrixRowsPerLane(inputIt->second), inputIt->second.shape[1]);
        layouts[exp.getResult()] = *resultLayout;
        eraseOps.push_back(operation);
        continue;
      }

      if (auto reciprocal = dyn_cast<nova::ReciprocalOp>(operation)) {
        auto inputIt = layouts.find(reciprocal.getInput());
        if (inputIt == layouts.end()) {
          reciprocal.emitOpError("requires lowered input metadata");
          signalPassFailure();
          return;
        }

        FailureOr<TensorLayout> resultLayout = propagateLayout(
            reciprocal, reciprocal.getResult().getType(), inputIt->second);
        if (failed(resultLayout)) {
          signalPassFailure();
          return;
        }
        createX1UnaryBankCommand<x1::ReciprocalOp>(
            builder, reciprocal, inputIt->second, *resultLayout,
            inputIt->second.shape[0], inputIt->second.shape[1]);
        layouts[reciprocal.getResult()] = *resultLayout;
        eraseOps.push_back(operation);
        continue;
      }

      if (auto matmul = dyn_cast<nova::MatmulOp>(operation)) {
        auto lhsIt = layouts.find(matmul.getLhs());
        auto rhsIt = layouts.find(matmul.getRhs());
        if (lhsIt == layouts.end() || rhsIt == layouts.end()) {
          matmul.emitOpError("requires lowered input metadata");
          signalPassFailure();
          return;
        }

        FailureOr<TensorLayout> resultLayout = propagateLayout(
            matmul, matmul.getResult().getType(), lhsIt->second);
        if (failed(resultLayout) ||
            failed(createX1Matmul(builder, matmul, lhsIt->second, rhsIt->second,
                                  *resultLayout))) {
          signalPassFailure();
          return;
        }
        layouts[matmul.getResult()] = *resultLayout;
        eraseOps.push_back(operation);
        continue;
      }

      if (auto barrier = dyn_cast<nova::BarrierOp>(operation)) {
        builder.create<x1::BarrierOp>(barrier.getLoc(), barrier.getModeAttr());
        eraseOps.push_back(operation);
        continue;
      }

      if (auto store = dyn_cast<nova::StoreOp>(operation)) {
        auto valueIt = layouts.find(store.getValue());
        if (valueIt == layouts.end()) {
          store.emitOpError("requires lowered input metadata");
          signalPassFailure();
          return;
        }

        FailureOr<SmallVector<int64_t>> start =
            toIntVector(store, store.getStartAttr(), "start");
        if (failed(start)) {
          signalPassFailure();
          return;
        }
        if (start->size() != valueIt->second.shape.size()) {
          store.emitOpError("requires start rank to match tensor rank");
          signalPassFailure();
          return;
        }

        MemRefLayout layout{valueIt->second, std::move(*start)};
        createX1Store(builder, store.getLoc(), store.getDest(), layout);
        eraseOps.push_back(operation);
        continue;
      }
    }

    for (Operation *operation : llvm::reverse(eraseOps))
      operation->erase();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::nova::createNovaToX1Pass() {
  return std::make_unique<NovaToX1Pass>();
}
