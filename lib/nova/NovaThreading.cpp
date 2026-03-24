#include "nova/NovaPasses.h"
#include "nova/NovaOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/MathExtras.h"

namespace mlir::nova {
#define GEN_PASS_DEF_NOVATHREADING
#include "nova/NovaPasses.h.inc"
} // namespace mlir::nova

using namespace mlir;
using namespace mlir::nova;

namespace {
static TensorLayoutAttr getTensorLayout(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType ? dyn_cast_or_null<TensorLayoutAttr>(tensorType.getEncoding())
                    : TensorLayoutAttr();
}

static std::optional<SmallVector<int64_t>> getLayoutArray(DenseI64ArrayAttr attr) {
  if (!attr)
    return std::nullopt;
  return SmallVector<int64_t>(attr.asArrayRef().begin(), attr.asArrayRef().end());
}

static DenseI64ArrayAttr getI64ArrayAttr(MLIRContext *context, ArrayRef<int64_t> values) {
  return DenseI64ArrayAttr::get(context, values);
}

static std::optional<int64_t> getThreadRows(Type type) {
  TensorLayoutAttr layout = getTensorLayout(type);
  std::optional<SmallVector<int64_t>> shape0 =
      getLayoutArray(layout ? layout.getShape0() : DenseI64ArrayAttr());
  if (!shape0 || shape0->empty())
    return std::nullopt;
  return (*shape0)[0];
}

static std::optional<int64_t> getSecondSliceRows(Type type) {
  TensorLayoutAttr layout = getTensorLayout(type);
  std::optional<SmallVector<int64_t>> shape1 =
      getLayoutArray(layout ? layout.getShape1() : DenseI64ArrayAttr());
  if (!shape1 || shape1->empty())
    return std::nullopt;
  return (*shape1)[0];
}

static int64_t getReduceAxis(nova::ReduceOp op) {
  return static_cast<int64_t>(op.getAxis());
}

static RankedTensorType withSliceMetadata(RankedTensorType type,
                                          std::optional<int64_t> firstRows) {
  MLIRContext *context = type.getContext();
  TensorLayoutAttr layout = dyn_cast_or_null<TensorLayoutAttr>(type.getEncoding());
  IntegerAttr bank0 = layout ? layout.getBank0() : IntegerAttr();
  IntegerAttr bank1 = layout ? layout.getBank1() : IntegerAttr();
  IntegerAttr space = layout ? layout.getSpace() : IntegerAttr();
  DenseI64ArrayAttr start0;
  DenseI64ArrayAttr start1;
  DenseI64ArrayAttr shape0;
  DenseI64ArrayAttr shape1;

  if (type.hasStaticShape() && type.getRank() == 2) {
    int64_t rows = type.getShape()[0];
    int64_t cols = type.getShape()[1];
    int64_t slice0Rows = firstRows ? std::min(rows, *firstRows) : rows;
    int64_t secondRows = firstRows ? std::max<int64_t>(rows - slice0Rows, 0) : 0;

    start0 = getI64ArrayAttr(context, {0, 0});
    shape0 = getI64ArrayAttr(context, {slice0Rows, cols});
    if (firstRows && secondRows > 0) {
      start1 = getI64ArrayAttr(context, {slice0Rows, 0});
      shape1 = getI64ArrayAttr(context, {secondRows, cols});
    }
  }

  return RankedTensorType::get(type.getShape(), type.getElementType(),
                               TensorLayoutAttr::get(context, bank0, bank1, start0,
                                                     start1, shape0, shape1, space));
}

static RankedTensorType withThreading(RankedTensorType type, int64_t threading) {
  return withSliceMetadata(type, threading);
}

static RankedTensorType withoutThreading(RankedTensorType type) {
  return withSliceMetadata(type, std::nullopt);
}

class NovaThreadingPass
    : public mlir::nova::impl::NovaThreadingBase<NovaThreadingPass> {
public:
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    auto propagateThreading = [&](Value input, Value result) {
      std::optional<int64_t> firstRows = getThreadRows(input.getType());
      if (!firstRows)
        return;

      auto resultType = dyn_cast<RankedTensorType>(result.getType());
      if (!resultType)
        return;
      result.setType(withThreading(resultType, *firstRows));
    };

    WalkResult walkResult = func.walk([&](Operation *op) -> WalkResult {
      if (auto load = dyn_cast<nova::LoadOp>(op)) {
        auto resultType = dyn_cast<RankedTensorType>(load.getResult().getType());
        if (!resultType || !resultType.hasStaticShape() ||
          resultType.getRank() < 1)
          return WalkResult::advance();

        int64_t threading = llvm::divideCeil(resultType.getShape()[0], 2LL);
        load.getResult().setType(withThreading(resultType, threading));
        return WalkResult::advance();
      }

      if (isa<nova::SquareOp, nova::ExpOp, nova::ReciprocalOp,
              nova::RsqrtOp>(op)) {
        propagateThreading(op->getOperand(0), op->getResult(0));
        return WalkResult::advance();
      }
      if (auto itof = dyn_cast<nova::IToFOp>(op)) {
        propagateThreading(itof.getInput(), itof.getResult());
        return WalkResult::advance();
      }
      if (auto ftoi = dyn_cast<nova::FToIOp>(op)) {
        propagateThreading(ftoi.getInput(), ftoi.getResult());
        return WalkResult::advance();
      }
      if (auto scalar = dyn_cast<nova::ScalarOp>(op)) {
        propagateThreading(scalar.getInput(), scalar.getResult());
        return WalkResult::advance();
      }
      if (auto scalarFma = dyn_cast<nova::ScalarFmaOp>(op)) {
        propagateThreading(scalarFma.getInput(), scalarFma.getResult());
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<nova::ReduceOp>(op)) {
        if (getReduceAxis(reduce) == 1)
          propagateThreading(reduce.getInput(), reduce.getResult());
        return WalkResult::advance();
      }
        if (auto matmul = dyn_cast<nova::MatmulOp>(op)) {
        if (std::optional<int64_t> lhsThreadRows =
                getThreadRows(matmul.getLhs().getType())) {
          auto resultType = dyn_cast<RankedTensorType>(matmul.getResult().getType());
          if (resultType)
            matmul.getResult().setType(withThreading(resultType, *lhsThreadRows));
        }

        auto stripThreading = [&](Value value) {
          auto result = dyn_cast<OpResult>(value);
          if (!result)
            return;

          auto type = dyn_cast<RankedTensorType>(result.getType());
          if (!type || !getThreadRows(type))
            return;
          result.setType(withoutThreading(type));
        };

        stripThreading(matmul.getRhs());
        stripThreading(matmul.getScale());
        stripThreading(matmul.getBias());
        return WalkResult::advance();
      }

      auto propagateBroadcastThreading = [&](Value lhs, Value rhs, Value result)
          -> WalkResult {
        std::optional<int64_t> lhsFirstRows = getThreadRows(lhs.getType());
        std::optional<int64_t> rhsFirstRows = getThreadRows(rhs.getType());
        if (!lhsFirstRows && !rhsFirstRows)
          return WalkResult::advance();

        auto resultType = dyn_cast<RankedTensorType>(result.getType());
        if (!resultType)
          return WalkResult::advance();
        if (!lhsFirstRows)
          result.setType(withThreading(resultType, *rhsFirstRows));
        else if (!rhsFirstRows)
          result.setType(withThreading(resultType, *lhsFirstRows));
        else
          result.setType(
              withThreading(resultType, std::max(*lhsFirstRows, *rhsFirstRows)));
        return WalkResult::advance();
      };

      auto propagateElementwiseThreading = [&](Value lhs, Value rhs,
                                               Value result) -> WalkResult {
        std::optional<int64_t> lhsFirstRows = getThreadRows(lhs.getType());
        std::optional<int64_t> rhsFirstRows = getThreadRows(rhs.getType());
        if (!lhsFirstRows || !rhsFirstRows)
          return WalkResult::advance();
        std::optional<int64_t> lhsSecondRows = getSecondSliceRows(lhs.getType());
        std::optional<int64_t> rhsSecondRows = getSecondSliceRows(rhs.getType());
        if (*lhsFirstRows != *rhsFirstRows || lhsSecondRows != rhsSecondRows) {
          op->emitOpError("lhs and rhs slice metadata must match");
          signalPassFailure();
          return WalkResult::interrupt();
        }

        auto resultType = dyn_cast<RankedTensorType>(result.getType());
        if (!resultType)
          return WalkResult::advance();
        result.setType(withThreading(resultType, *lhsFirstRows));
        return WalkResult::advance();
      };

      if (auto broadcast = dyn_cast<nova::BroadcastOp>(op))
        return propagateBroadcastThreading(broadcast.getLhs(),
                                           broadcast.getRhs(),
                                           broadcast.getResult());
      if (auto elementwise = dyn_cast<nova::ElementwiseOp>(op))
        return propagateElementwiseThreading(elementwise.getLhs(),
                                             elementwise.getRhs(),
                                             elementwise.getResult());

      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted())
      return;
  }
};
} // namespace

std::unique_ptr<Pass> mlir::nova::createNovaThreadingPass() {
  return std::make_unique<NovaThreadingPass>();
}
