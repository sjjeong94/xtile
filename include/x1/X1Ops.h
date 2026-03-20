#ifndef X1_X1OPS_H
#define X1_X1OPS_H

#include "x1/X1Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "x1/X1Ops.h.inc"

#endif // X1_X1OPS_H
