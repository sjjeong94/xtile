#ifndef XT_XTPASSES_H
#define XT_XTPASSES_H

#include "xt/XTDialect.h"
#include "xt/XTOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::xt {

std::unique_ptr<Pass> createXTToNovaPass();
std::unique_ptr<Pass> createXTLivenessPass();

#define GEN_PASS_DECL
#include "xt/XTPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "xt/XTPasses.h.inc"

} // namespace mlir::xt

#endif // XT_XTPASSES_H
