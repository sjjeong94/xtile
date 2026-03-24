#include "nova/NovaDialect.h"
#include "nova/NovaOps.h"

#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::nova;

#define GET_ATTRDEF_CLASSES
#include "nova/NovaAttrDefs.cpp.inc"
#include "nova/NovaOpsDialect.cpp.inc"

namespace {
static void printLayoutArray(AsmPrinter &printer, DenseI64ArrayAttr attr) {
  printer << '[';
  llvm::interleaveComma(attr.asArrayRef(), printer,
                        [&](int64_t value) { printer << value; });
  printer << ']';
}

static FailureOr<DenseI64ArrayAttr> parseLayoutArray(AsmParser &parser) {
  SmallVector<int64_t> values;
  if (parser.parseLSquare())
    return failure();
  if (succeeded(parser.parseOptionalRSquare()))
    return DenseI64ArrayAttr::get(parser.getContext(), values);

  do {
    int64_t value;
    if (parser.parseInteger(value))
      return failure();
    values.push_back(value);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRSquare())
    return failure();
  return DenseI64ArrayAttr::get(parser.getContext(), values);
}

} // namespace

Attribute TensorLayoutAttr::parse(AsmParser &parser, Type) {
  IntegerAttr bank0;
  IntegerAttr bank1;
  DenseI64ArrayAttr start0;
  DenseI64ArrayAttr start1;
  DenseI64ArrayAttr shape0;
  DenseI64ArrayAttr shape1;
  IntegerAttr space;

  auto parseIntegerField = [&](IntegerAttr &target) -> ParseResult {
    int64_t value;
    if (parser.parseEqual() || parser.parseInteger(value))
      return failure();
    target = IntegerAttr::get(IntegerType::get(parser.getContext(), 64), value);
    return success();
  };
  auto parseArrayField = [&](DenseI64ArrayAttr &target) -> ParseResult {
    if (parser.parseEqual())
      return failure();
    FailureOr<DenseI64ArrayAttr> attr = parseLayoutArray(parser);
    if (failed(attr))
      return failure();
    target = *attr;
    return success();
  };
  auto parseRangeField = [&](DenseI64ArrayAttr &start,
                             DenseI64ArrayAttr &shape) -> ParseResult {
    FailureOr<DenseI64ArrayAttr> parsedStart = parseLayoutArray(parser);
    if (failed(parsedStart))
      return failure();
    FailureOr<DenseI64ArrayAttr> parsedShape = parseLayoutArray(parser);
    if (failed(parsedShape))
      return failure();
    start = *parsedStart;
    shape = *parsedShape;
    return success();
  };

  if (parser.parseLess())
    return {};
  if (succeeded(parser.parseOptionalGreater()))
    return TensorLayoutAttr::get(parser.getContext(), bank0, bank1, start0,
                                 start1, shape0, shape1, space);

  while (true) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword))
      return {};

    if (keyword == "range0") {
      if (parseRangeField(start0, shape0))
        return {};
    } else if (keyword == "range1") {
      if (parseRangeField(start1, shape1))
        return {};
    } else if (keyword == "bank0") {
      if (parseIntegerField(bank0))
        return {};
    } else if (keyword == "bank1") {
      if (parseIntegerField(bank1))
        return {};
    } else if (keyword == "start0") {
      if (parseArrayField(start0))
        return {};
    } else if (keyword == "start1") {
      if (parseArrayField(start1))
        return {};
    } else if (keyword == "shape0") {
      if (parseArrayField(shape0))
        return {};
    } else if (keyword == "shape1") {
      if (parseArrayField(shape1))
        return {};
    } else if (keyword == "space") {
      if (parseIntegerField(space))
        return {};
    } else {
      parser.emitError(parser.getCurrentLocation(),
                       "unknown nova tensor layout field");
      return {};
    }

    if (succeeded(parser.parseOptionalGreater()))
      break;
    if (parser.parseComma())
      return {};
  }

  return TensorLayoutAttr::get(parser.getContext(), bank0, bank1, start0,
                               start1, shape0, shape1, space);
}

void TensorLayoutAttr::print(AsmPrinter &printer) const {
  printer << '<';
  bool first = true;
  auto printSeparator = [&]() {
    if (!first)
      printer << ", ";
    first = false;
  };
  auto printIntegerField = [&](StringRef name, IntegerAttr attr) {
    if (!attr)
      return;
    printSeparator();
    printer << name << " = " << attr.getInt();
  };
  auto printRangeField = [&](StringRef name, DenseI64ArrayAttr start,
                             DenseI64ArrayAttr shape) {
    if (!start || !shape)
      return;
    printSeparator();
    printer << name << ' ';
    printLayoutArray(printer, start);
    printer << ' ';
    printLayoutArray(printer, shape);
  };

  printRangeField("range0", getStart0(), getShape0());
  printRangeField("range1", getStart1(), getShape1());
  printIntegerField("bank0", getBank0());
  printIntegerField("bank1", getBank1());
  printIntegerField("space", getSpace());
  printer << '>';
}

void NovaDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "nova/NovaAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "nova/NovaOps.cpp.inc"
      >();
}
