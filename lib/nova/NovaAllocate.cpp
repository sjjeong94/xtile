#include "nova/NovaPasses.h"
#include "nova/NovaOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <optional>

namespace mlir::nova {
#define GEN_PASS_DEF_NOVAALLOCATE
#include "nova/NovaPasses.h.inc"
} // namespace mlir::nova

using namespace mlir;

namespace {
constexpr int64_t kBankSize = 2048 * 64;
constexpr int64_t kBankSize2 = kBankSize * 2;

enum class MemorySpace : int64_t {
  DRAM_INPUT = 0,
  DRAM_OUTPUT = 1,
  DRAM_WEIGHT = 2,
  SRAM = 3,
  RF_MUL = 4,
  RF_ADD = 5,
  RF_SMUL = 6,
  RF_SADD = 7,
  RF_SE = 8,
  RF_GE = 9,
};

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

struct ResultProperties {
  std::optional<MemorySpace> space;
  bool eligibleForAllocation;
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

static bool isAllocationEligibleType(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || !tensorType.hasStaticShape())
    return false;
  return !isSingleElementTensor(tensorType);
}

static std::optional<MemorySpace> getSpaceAssignment(OpResult value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || isSingleElementTensor(type))
    return std::nullopt;

  std::optional<MemorySpace> specialSpace;
  for (OpOperand &use : value.getUses()) {
    auto matmulOp = dyn_cast<nova::MatmulOp>(use.getOwner());
    if (!matmulOp)
      continue;

    std::optional<MemorySpace> candidate;
    switch (use.getOperandNumber()) {
    case 2:
      candidate = MemorySpace::RF_MUL;
      break;
    case 3:
      candidate = MemorySpace::RF_ADD;
      break;
    default:
      break;
    }

    if (!candidate)
      continue;
    if (specialSpace && *specialSpace != *candidate)
      return MemorySpace::SRAM;
    specialSpace = candidate;
  }

  return specialSpace.value_or(MemorySpace::SRAM);
}

static ResultProperties getResultProperties(OpResult value) {
  auto space = getSpaceAssignment(value);
  return ResultProperties{space, space && *space == MemorySpace::SRAM};
}

static RankedTensorType withEncoding(RankedTensorType type,
                                     std::optional<int64_t> bank,
                                     std::optional<MemorySpace> space) {
  MLIRContext *context = type.getContext();
  NamedAttrList attrs;
  if (auto dict = dyn_cast_or_null<DictionaryAttr>(type.getEncoding()))
    attrs.append(dict.getValue());
  if (bank) {
    attrs.set("bank", IntegerAttr::get(IntegerType::get(context, 64), *bank));
  }
  if (space) {
    attrs.set("space",
              IntegerAttr::get(IntegerType::get(context, 64),
                               static_cast<int64_t>(*space)));
  }
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
    DenseMap<Value, ResultProperties> resultProperties;

    for (Operation *op : ops) {
      for (OpResult result : op->getResults()) {
        if (!isAllocationEligibleType(result.getType()))
          continue;

        ResultProperties properties = getResultProperties(result);
        resultProperties[result] = properties;
        if (!properties.eligibleForAllocation)
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

    for (Operation *op : ops) {
      for (OpResult result : op->getResults()) {
        auto it = resultProperties.find(result);
        if (it == resultProperties.end() || it->second.eligibleForAllocation)
          continue;

        auto type = cast<RankedTensorType>(result.getType());
        result.setType(withEncoding(type, std::nullopt, it->second.space));
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
        value.setType(withEncoding(type, *addr / kBankSize,
                                   resultProperties[value].space));
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
