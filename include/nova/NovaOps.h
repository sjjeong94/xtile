#ifndef NOVA_NOVAOPS_H
#define NOVA_NOVAOPS_H

#include "nova/NovaDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "nova/NovaOps.h.inc"

#endif // NOVA_NOVAOPS_H
