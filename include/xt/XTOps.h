#ifndef XT_XTOPS_H
#define XT_XTOPS_H

#include "xt/XTDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/IR/PatternMatch.h"

#define GET_OP_CLASSES
#include "xt/XTOps.h.inc"

#endif // XT_XTOPS_H
