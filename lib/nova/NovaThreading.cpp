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

namespace {
static std::optional<DictionaryAttr> getEncoding(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType)
    return std::nullopt;

  auto encoding = dyn_cast_or_null<DictionaryAttr>(tensorType.getEncoding());
  if (!encoding)
    return std::nullopt;
  return encoding;
}

static std::optional<SmallVector<int64_t>> getEncodingI64Array(Type type,
                                                               StringRef name) {
  std::optional<DictionaryAttr> encoding = getEncoding(type);
  if (!encoding)
    return std::nullopt;

  if (auto denseAttr = dyn_cast_or_null<DenseI64ArrayAttr>((*encoding).get(name)))
    return SmallVector<int64_t>(denseAttr.asArrayRef().begin(),
                                denseAttr.asArrayRef().end());
  if (auto attr = dyn_cast_or_null<ArrayAttr>((*encoding).get(name))) {
    SmallVector<int64_t> values;
    values.reserve(attr.size());
    for (Attribute element : attr) {
      auto intAttr = dyn_cast<IntegerAttr>(element);
      if (!intAttr)
        return std::nullopt;
      values.push_back(intAttr.getInt());
    }
    return values;
  }
  return std::nullopt;
}

static DenseI64ArrayAttr getI64ArrayAttr(MLIRContext *context, ArrayRef<int64_t> values) {
  return DenseI64ArrayAttr::get(context, values);
}

static std::optional<int64_t> getThreadRows(Type type) {
  std::optional<SmallVector<int64_t>> shape0 = getEncodingI64Array(type, "shape0");
  if (!shape0 || shape0->empty())
    return std::nullopt;
  return (*shape0)[0];
}

static std::optional<int64_t> getSecondSliceRows(Type type) {
  std::optional<SmallVector<int64_t>> shape1 = getEncodingI64Array(type, "shape1");
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
  NamedAttrList attrs;
  if (auto dict = dyn_cast_or_null<DictionaryAttr>(type.getEncoding()))
    attrs.append(dict.getValue());
  attrs.erase("threading");

  if (type.hasStaticShape() && type.getRank() == 2) {
    int64_t rows = type.getShape()[0];
    int64_t cols = type.getShape()[1];
    int64_t slice0Rows = firstRows ? std::min(rows, *firstRows) : rows;
    int64_t secondRows = firstRows ? std::max<int64_t>(rows - slice0Rows, 0) : 0;

    attrs.set("start0", getI64ArrayAttr(context, {0, 0}));
    attrs.set("shape0", getI64ArrayAttr(context, {slice0Rows, cols}));
    if (firstRows && secondRows > 0) {
      attrs.set("start1", getI64ArrayAttr(context, {slice0Rows, 0}));
      attrs.set("shape1", getI64ArrayAttr(context, {secondRows, cols}));
    } else {
      attrs.erase("start1");
      attrs.erase("shape1");
    }
  }

  return RankedTensorType::get(type.getShape(), type.getElementType(),
                               DictionaryAttr::get(context, attrs));
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
