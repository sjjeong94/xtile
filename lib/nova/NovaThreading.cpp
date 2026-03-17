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
static std::optional<int64_t> getThreading(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType)
    return std::nullopt;

  auto encoding = dyn_cast_or_null<DictionaryAttr>(tensorType.getEncoding());
  if (!encoding)
    return std::nullopt;

  auto attr = dyn_cast_or_null<IntegerAttr>(encoding.get("threading"));
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

static RankedTensorType withThreading(RankedTensorType type, int64_t threading) {
  MLIRContext *context = type.getContext();
  NamedAttrList attrs;
  if (auto dict = dyn_cast_or_null<DictionaryAttr>(type.getEncoding()))
    attrs.append(dict.getValue());
  attrs.set("threading",
            IntegerAttr::get(IntegerType::get(context, 64), threading));
  return RankedTensorType::get(type.getShape(), type.getElementType(),
                               DictionaryAttr::get(context, attrs));
}

static RankedTensorType withoutThreading(RankedTensorType type) {
  MLIRContext *context = type.getContext();
  NamedAttrList attrs;
  if (auto dict = dyn_cast_or_null<DictionaryAttr>(type.getEncoding()))
    attrs.append(dict.getValue());
  attrs.erase("threading");

  Attribute encoding;
  if (!attrs.empty())
    encoding = DictionaryAttr::get(context, attrs);
  return RankedTensorType::get(type.getShape(), type.getElementType(), encoding);
}

class NovaThreadingPass
    : public mlir::nova::impl::NovaThreadingBase<NovaThreadingPass> {
public:
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    auto propagateThreading = [&](Value input, Value result) {
      std::optional<int64_t> threading = getThreading(input.getType());
      if (!threading)
        return;

      auto resultType = dyn_cast<RankedTensorType>(result.getType());
      if (!resultType)
        return;
      result.setType(withThreading(resultType, *threading));
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

      if (auto square = dyn_cast<nova::SquareOp>(op)) {
        propagateThreading(square.getInput(), square.getResult());
        return WalkResult::advance();
      }
      if (auto rsqrt = dyn_cast<nova::RsqrtOp>(op)) {
        propagateThreading(rsqrt.getInput(), rsqrt.getResult());
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
        propagateThreading(reduce.getInput(), reduce.getResult());
        return WalkResult::advance();
      }
      if (auto matmul = dyn_cast<nova::MatmulOp>(op)) {
        if (std::optional<int64_t> lhsThreading =
                getThreading(matmul.getLhs().getType())) {
          auto resultType = dyn_cast<RankedTensorType>(matmul.getResult().getType());
          if (resultType)
            matmul.getResult().setType(withThreading(resultType, *lhsThreading));
        }

        auto stripThreading = [&](Value value) {
          auto result = dyn_cast<OpResult>(value);
          if (!result)
            return;

          auto type = dyn_cast<RankedTensorType>(result.getType());
          if (!type || !getThreading(type))
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
        std::optional<int64_t> lhsThreading = getThreading(lhs.getType());
        std::optional<int64_t> rhsThreading = getThreading(rhs.getType());
        if (!lhsThreading && !rhsThreading)
          return WalkResult::advance();
        if (!lhsThreading || !rhsThreading) {
          op->emitOpError("lhs and rhs threading must both be present");
          signalPassFailure();
          return WalkResult::interrupt();
        }

        auto resultType = dyn_cast<RankedTensorType>(result.getType());
        if (!resultType)
          return WalkResult::advance();
        result.setType(withThreading(resultType,
                                     std::max(*lhsThreading, *rhsThreading)));
        return WalkResult::advance();
      };

      auto propagateElementwiseThreading = [&](Value lhs, Value rhs,
                                               Value result) -> WalkResult {
        std::optional<int64_t> lhsThreading = getThreading(lhs.getType());
        std::optional<int64_t> rhsThreading = getThreading(rhs.getType());
        if (!lhsThreading || !rhsThreading)
          return WalkResult::advance();
        if (*lhsThreading != *rhsThreading) {
          op->emitOpError("lhs and rhs threading must match");
          signalPassFailure();
          return WalkResult::interrupt();
        }

        auto resultType = dyn_cast<RankedTensorType>(result.getType());
        if (!resultType)
          return WalkResult::advance();
        result.setType(withThreading(resultType, *lhsThreading));
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
