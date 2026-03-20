#ifndef XT_NOVAPASSES_H
#define XT_NOVAPASSES_H

#include "nova/NovaDialect.h"
#include "nova/NovaOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::nova {

std::unique_ptr<Pass> createNovaAllocatePass();
std::unique_ptr<Pass> createNovaBarrierPass();
std::unique_ptr<Pass> createNovaOptimizePass();
std::unique_ptr<Pass> createNovaThreadingPass();

#define GEN_PASS_DECL
#include "nova/NovaPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "nova/NovaPasses.h.inc"

} // namespace mlir::nova

#endif // XT_NOVAPASSES_H
