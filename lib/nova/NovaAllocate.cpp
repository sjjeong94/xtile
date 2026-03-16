#include "nova/NovaPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::nova {
#define GEN_PASS_DEF_NOVAALLOCATE
#include "nova/NovaPasses.h.inc"
} // namespace mlir::nova

using namespace mlir;

namespace {
constexpr int64_t kBankSize = 2048 * 64;
constexpr int64_t kBankSize2 = kBankSize * 2;

struct AllocationInfo {
  OpResult value;
  int64_t size;
  int64_t addr;
};

struct ValueLifetime {
  OpResult value;
  int64_t createdAt;
  int64_t deletedAt;
  int64_t size;
};

static bool isSingleElementTensor(RankedTensorType type) {
  if (!type || !type.hasStaticShape())
    return false;
  return type.getNumElements() == 1;
}

static FailureOr<int64_t> getElementByteWidth(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return llvm::divideCeil(intType.getWidth(), 8);
  if (auto floatType = dyn_cast<FloatType>(type))
    return llvm::divideCeil(floatType.getWidth(), 8);
  if (type.isIndex())
    return static_cast<int64_t>(sizeof(int64_t));
  return failure();
}

static FailureOr<int64_t> getTensorSizeInBytes(RankedTensorType type) {
  if (!type || !type.hasStaticShape())
    return failure();

  FailureOr<int64_t> elementBytes = getElementByteWidth(type.getElementType());
  if (failed(elementBytes))
    return failure();
  return type.getNumElements() * *elementBytes;
}

static int64_t alignToBank(int64_t addr) {
  return llvm::alignTo(addr, kBankSize2);
}

static std::pair<int64_t, int64_t> getBankRange(int64_t addr, int64_t size) {
  int64_t startBank = addr / kBankSize2;
  int64_t endBank = (addr + size - 1) / kBankSize2 + 1;
  return {startBank, endBank};
}

static int64_t getBankAlignedEnd(const AllocationInfo &info) {
  auto [_, endBank] = getBankRange(info.addr, info.size);
  return endBank * kBankSize2;
}

static bool isEligibleForAllocation(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || !tensorType.hasStaticShape())
    return false;
  return !isSingleElementTensor(tensorType);
}

static RankedTensorType withAddrEncoding(RankedTensorType type, int64_t addr) {
  MLIRContext *context = type.getContext();
  NamedAttrList attrs;
  if (auto dict = dyn_cast_or_null<DictionaryAttr>(type.getEncoding()))
    attrs.append(dict.getValue());
  int64_t bank = addr / kBankSize;
  attrs.set("bank", IntegerAttr::get(IntegerType::get(context, 64), bank));
  return RankedTensorType::get(type.getShape(), type.getElementType(),
                               DictionaryAttr::get(context, attrs));
}

class NovaAllocatePass
    : public mlir::nova::impl::NovaAllocateBase<NovaAllocatePass> {
public:
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (!func.getBody().hasOneBlock()) {
      func.emitOpError("nova-allocate only supports single-block functions");
      signalPassFailure();
      return;
    }

    Block &block = func.getBody().front();
    SmallVector<Operation *> ops;
    ops.reserve(block.getOperations().size());
    for (Operation &op : block)
      ops.push_back(&op);

    DenseMap<Operation *, int64_t> opIndices;
    for (auto [index, op] : llvm::enumerate(ops))
      opIndices[op] = index;

    SmallVector<SmallVector<OpResult>> createdAt(ops.size());
    SmallVector<SmallVector<OpResult>> deletedAt(ops.size() + 1);
    SmallVector<ValueLifetime> lifetimes;

    for (Operation *op : ops) {
      for (OpResult result : op->getResults()) {
        if (!isEligibleForAllocation(result.getType()))
          continue;

        auto tensorType = cast<RankedTensorType>(result.getType());
        FailureOr<int64_t> size = getTensorSizeInBytes(tensorType);
        if (failed(size)) {
          op->emitOpError("nova-allocate requires tensor element types with a "
                          "known byte width");
          signalPassFailure();
          return;
        }

        int64_t created = opIndices[op];
        int64_t deleted = created + 1;
        for (Operation *user : result.getUsers())
          deleted = std::max(deleted, opIndices[user] + 1);

        createdAt[created].push_back(result);
        deletedAt[deleted].push_back(result);
        lifetimes.push_back(ValueLifetime{result, created, deleted, *size});
      }
    }

    DenseMap<Value, ValueLifetime> lifetimeByValue;
    for (const ValueLifetime &lifetime : lifetimes)
      lifetimeByValue[lifetime.value] = lifetime;

    SmallVector<AllocationInfo> liveAllocs;
    auto sortLiveAllocs = [&]() {
      llvm::sort(liveAllocs,
                 [](const AllocationInfo &lhs, const AllocationInfo &rhs) {
                   return lhs.addr < rhs.addr;
                 });
    };

    auto deallocateValue = [&](OpResult value) {
      llvm::erase_if(liveAllocs, [&](const AllocationInfo &info) {
        return info.value == value;
      });
    };

    auto allocateValue = [&](OpResult value) -> FailureOr<int64_t> {
      const ValueLifetime &lifetime = lifetimeByValue[value];
      int64_t start = 0;
      sortLiveAllocs();
      for (const AllocationInfo &info : liveAllocs) {
        int64_t candidate = alignToBank(start);
        if (candidate + lifetime.size <= info.addr)
          break;
        start = getBankAlignedEnd(info);
      }
      start = alignToBank(start);
      liveAllocs.push_back(AllocationInfo{value, lifetime.size, start});
      sortLiveAllocs();
      return start;
    };

    for (size_t index = 0; index < ops.size(); ++index) {
      for (OpResult value : deletedAt[index])
        deallocateValue(value);

      for (OpResult value : createdAt[index]) {
        FailureOr<int64_t> addr = allocateValue(value);
        if (failed(addr)) {
          value.getOwner()->emitOpError("nova-allocate failed to assign an "
                                        "address");
          signalPassFailure();
          return;
        }

        auto type = cast<RankedTensorType>(value.getType());
        value.setType(withAddrEncoding(type, *addr));
      }
    }

    for (OpResult value : deletedAt[ops.size()])
      deallocateValue(value);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::nova::createNovaAllocatePass() {
  return std::make_unique<NovaAllocatePass>();
}
