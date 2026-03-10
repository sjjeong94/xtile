#include "xt/XTOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::xt;

#define GET_OP_CLASSES
#include "xt/XTOps.cpp.inc"

static LogicalResult verifyTileAttr(Operation *op, DenseI64ArrayAttr tile,
                                    RankedTensorType tensorType) {
  if (!tile || tile.empty())
    return op->emitOpError("requires tile attribute with at least one entry");
  if (!tensorType.hasStaticShape())
    return op->emitOpError("requires a statically shaped tensor");
  if (static_cast<int64_t>(tile.size()) != tensorType.getRank())
    return op->emitOpError("tile rank must match tensor rank");
  for (int64_t i = 0, e = tensorType.getRank(); i < e; ++i) {
    if (tile[i] <= 0)
      return op->emitOpError("requires positive tile dimensions");
    if (tensorType.getDimSize(i) != tile[i])
      return op->emitOpError("tile attribute must match tensor shape");
  }
  return success();
}

static LogicalResult verifyMemRefAndTensor(Operation *op, MemRefType memRefType,
                                           RankedTensorType tensorType) {
  if (!memRefType)
    return op->emitOpError("requires a ranked memref operand");
  if (memRefType.getRank() != tensorType.getRank())
    return op->emitOpError("memref rank must match tensor rank");
  if (memRefType.getElementType() != tensorType.getElementType())
    return op->emitOpError("requires matching memref/tensor element types");
  return success();
}

static ParseResult parseSingleTypeResultOp(OpAsmParser &parser,
                                           OperationState &result,
                                           SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
                                           int32_t operandCount) {
  Type type;
  if (parser.parseLParen() || parser.parseOperandList(operands, operandCount) ||
      parser.parseRParen() || parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return failure();
  if (parser.resolveOperands(operands, type, parser.getNameLoc(), result.operands))
    return failure();
  result.addTypes(type);
  return success();
}

static ParseResult parseTileAttr(OpAsmParser &parser,
                                 OperationState &result) {
  Builder &builder = parser.getBuilder();
  SmallVector<int64_t> tileDims;
  if (parser.parseLBrace() || parser.parseKeyword("tile") || parser.parseEqual() ||
      parser.parseLSquare())
    return failure();
  do {
    int64_t dim;
    if (parser.parseInteger(dim))
      return failure();
    tileDims.push_back(dim);
  } while (succeeded(parser.parseOptionalComma()));
  if (parser.parseRSquare() || parser.parseRBrace())
    return failure();
  result.addAttribute("tile", builder.getDenseI64ArrayAttr(tileDims));
  return success();
}

static void printTileAttr(OpAsmPrinter &printer, DenseI64ArrayAttr tile) {
  printer << " {tile = [";
  llvm::interleaveComma(tile.asArrayRef(), printer,
                        [&](int64_t dim) { printer << dim; });
  printer << "]}";
}

static void printSingleTypeResultOp(OpAsmPrinter &printer, Operation *op) {
  printer << "(";
  printer.printOperands(op->getOperands());
  printer << ")";
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : " << op->getResult(0).getType();
}

ParseResult GetTileBlockIdOp::parse(OpAsmParser &parser, OperationState &result) {
  Type type;
  if (parser.parseLParen() || parser.parseRParen() ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColonType(type))
    return failure();
  result.addTypes({type, type, type});
  return success();
}

void GetTileBlockIdOp::print(OpAsmPrinter &printer) {
  printer << "()";
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getX().getType();
}

ParseResult LoadOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand source;
  SmallVector<OpAsmParser::UnresolvedOperand> coords;
  MemRefType sourceType;
  Type resultType;
  Type coordType = parser.getBuilder().getI32Type();
  if (parser.parseLParen() || parser.parseOperand(source) ||
      parser.parseTrailingOperandList(coords) || parser.parseRParen() ||
      parseTileAttr(parser, result) ||
      parser.parseColonType(sourceType) || parser.parseArrow() ||
      parser.parseType(resultType))
    return failure();
  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperands(coords, coordType, result.operands))
    return failure();
  result.addTypes(resultType);
  return success();
}

void LoadOp::print(OpAsmPrinter &printer) {
  printer << "(" << getSource();
  for (Value coord : getCoords())
    printer << ", " << coord;
  printer << ")";
  printTileAttr(printer, getTileAttr());
  printer << " : " << getSource().getType() << " -> " << getResult().getType();
}

ParseResult StoreOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand value;
  OpAsmParser::UnresolvedOperand dest;
  SmallVector<OpAsmParser::UnresolvedOperand> coords;
  Type valueType;
  MemRefType destType;
  Type coordType = parser.getBuilder().getI32Type();
  if (parser.parseLParen() || parser.parseOperand(value) ||
      parser.parseComma() || parser.parseOperand(dest) ||
      parser.parseTrailingOperandList(coords) || parser.parseRParen() ||
      parseTileAttr(parser, result) ||
      parser.parseColonType(valueType) || parser.parseArrow() ||
      parser.parseType(destType))
    return failure();
  if (parser.resolveOperand(value, valueType, result.operands) ||
      parser.resolveOperand(dest, destType, result.operands) ||
      parser.resolveOperands(coords, coordType, result.operands))
    return failure();
  return success();
}

void StoreOp::print(OpAsmPrinter &printer) {
  printer << "(" << getValue() << ", " << getDest();
  for (Value coord : getCoords())
    printer << ", " << coord;
  printer << ")";
  printTileAttr(printer, getTileAttr());
  printer << " : " << getValue().getType() << " -> " << getDest().getType();
}

ParseResult AddOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  return parseSingleTypeResultOp(parser, result, operands, 2);
}

void AddOp::print(OpAsmPrinter &printer) { printSingleTypeResultOp(printer, *this); }

ParseResult ExpOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  return parseSingleTypeResultOp(parser, result, operands, 1);
}

void ExpOp::print(OpAsmPrinter &printer) { printSingleTypeResultOp(printer, *this); }

LogicalResult LoadOp::verify() {
  auto tensorType = llvm::dyn_cast<RankedTensorType>(getResult().getType());
  auto memRefType = llvm::dyn_cast<MemRefType>(getSource().getType());
  if (!tensorType)
    return emitOpError("requires a ranked tensor result");
  if (getCoords().empty())
    return emitOpError("requires at least one coordinate");
  if (failed(verifyTileAttr(*this, getTileAttr(), tensorType)))
    return failure();
  if (static_cast<int64_t>(getCoords().size()) != tensorType.getRank())
    return emitOpError("coordinate count must match tile rank");
  return verifyMemRefAndTensor(*this, memRefType, tensorType);
}

LogicalResult StoreOp::verify() {
  auto tensorType = llvm::dyn_cast<RankedTensorType>(getValue().getType());
  auto memRefType = llvm::dyn_cast<MemRefType>(getDest().getType());
  if (!tensorType)
    return emitOpError("requires a ranked tensor operand");
  if (getCoords().empty())
    return emitOpError("requires at least one coordinate");
  if (failed(verifyTileAttr(*this, getTileAttr(), tensorType)))
    return failure();
  if (static_cast<int64_t>(getCoords().size()) != tensorType.getRank())
    return emitOpError("coordinate count must match tile rank");
  return verifyMemRefAndTensor(*this, memRefType, tensorType);
}

LogicalResult AddOp::verify() {
  auto lhsType = llvm::dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsType = llvm::dyn_cast<RankedTensorType>(getRhs().getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(getResult().getType());
  if (!lhsType || !rhsType || !resultType)
    return emitOpError("requires ranked tensor operands and result");
  if (lhsType != rhsType || lhsType != resultType)
    return emitOpError("requires operand and result tensor types to match");
  if (!lhsType.hasStaticShape())
    return emitOpError("requires statically shaped tensors");
  return success();
}

LogicalResult ExpOp::verify() {
  auto inputType = llvm::dyn_cast<RankedTensorType>(getInput().getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(getResult().getType());
  if (!inputType || !resultType)
    return emitOpError("requires ranked tensor operand and result");
  if (inputType != resultType)
    return emitOpError("requires operand and result tensor types to match");
  if (!inputType.hasStaticShape())
    return emitOpError("requires statically shaped tensors");
  return success();
}

static bool isZeroTensor(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  auto dense = llvm::dyn_cast<DenseElementsAttr>(constant.getValue());
  if (!dense || !dense.isSplat())
    return false;
  auto splat = dense.getSplatValue<Attribute>();
  if (auto floatAttr = llvm::dyn_cast<FloatAttr>(splat))
    return floatAttr.getValue().isZero();
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(splat))
    return intAttr.getValue().isZero();
  return false;
}

namespace {
struct FoldAddZeroPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (isZeroTensor(op.getLhs())) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }
    if (isZeroTensor(op.getRhs())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    return failure();
  }
};
} // namespace

void AddOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<FoldAddZeroPattern>(context);
}
