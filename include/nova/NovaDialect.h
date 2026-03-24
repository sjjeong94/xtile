#ifndef NOVA_NOVADIALECT_H
#define NOVA_NOVADIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"

#include "nova/NovaOpsDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "nova/NovaAttrDefs.h.inc"

#endif // NOVA_NOVADIALECT_H
