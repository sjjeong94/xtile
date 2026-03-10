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
  if (!tile || tile.size() != 2)
    return op->emitOpError("requires tile attribute with exactly two entries");
  if (tensorType.getRank() != 2 || !tensorType.hasStaticShape())
    return op->emitOpError("requires a statically shaped rank-2 tensor");
  if (tile[0] <= 0 || tile[1] <= 0)
    return op->emitOpError("requires positive tile dimensions");
  if (tensorType.getDimSize(0) != tile[0] || tensorType.getDimSize(1) != tile[1])
    return op->emitOpError("tile attribute must match tensor shape");
  return success();
}

static LogicalResult verifyMemRefAndTensor(Operation *op, MemRefType memRefType,
                                           RankedTensorType tensorType) {
  if (!memRefType)
    return op->emitOpError("requires a ranked memref operand");
  if (memRefType.getRank() != 2)
    return op->emitOpError("requires a rank-2 memref");
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
  int64_t tileM;
  int64_t tileN;
  if (parser.parseLBrace() || parser.parseKeyword("tile") || parser.parseEqual() ||
      parser.parseLSquare() || parser.parseInteger(tileM) ||
      parser.parseComma() || parser.parseInteger(tileN) ||
      parser.parseRSquare() || parser.parseRBrace())
    return failure();
  result.addAttribute("tile", builder.getDenseI64ArrayAttr({tileM, tileN}));
  return success();
}

static void printTileAttr(OpAsmPrinter &printer, DenseI64ArrayAttr tile) {
  printer << " {tile = [" << tile[0] << ", " << tile[1] << "]}";
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
  OpAsmParser::UnresolvedOperand tileX;
  OpAsmParser::UnresolvedOperand tileY;
  MemRefType sourceType;
  Type resultType;
  Type coordType = parser.getBuilder().getI32Type();
  if (parser.parseLParen() || parser.parseOperand(source) ||
      parser.parseComma() || parser.parseOperand(tileX) ||
      parser.parseComma() || parser.parseOperand(tileY) ||
      parser.parseRParen() || parseTileAttr(parser, result) ||
      parser.parseColonType(sourceType) || parser.parseArrow() ||
      parser.parseType(resultType))
    return failure();
  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperand(tileX, coordType, result.operands) ||
      parser.resolveOperand(tileY, coordType, result.operands))
    return failure();
  result.addTypes(resultType);
  return success();
}

void LoadOp::print(OpAsmPrinter &printer) {
  printer << "(" << getSource() << ", " << getTileX() << ", " << getTileY() << ")";
  printTileAttr(printer, getTileAttr());
  printer << " : " << getSource().getType() << " -> " << getResult().getType();
}

ParseResult StoreOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand value;
  OpAsmParser::UnresolvedOperand dest;
  OpAsmParser::UnresolvedOperand tileX;
  OpAsmParser::UnresolvedOperand tileY;
  Type valueType;
  MemRefType destType;
  Type coordType = parser.getBuilder().getI32Type();
  if (parser.parseLParen() || parser.parseOperand(value) ||
      parser.parseComma() || parser.parseOperand(dest) ||
      parser.parseComma() || parser.parseOperand(tileX) ||
      parser.parseComma() || parser.parseOperand(tileY) ||
      parser.parseRParen() || parseTileAttr(parser, result) ||
      parser.parseColonType(valueType) || parser.parseArrow() ||
      parser.parseType(destType))
    return failure();
  if (parser.resolveOperand(value, valueType, result.operands) ||
      parser.resolveOperand(dest, destType, result.operands) ||
      parser.resolveOperand(tileX, coordType, result.operands) ||
      parser.resolveOperand(tileY, coordType, result.operands))
    return failure();
  return success();
}

void StoreOp::print(OpAsmPrinter &printer) {
  printer << "(" << getValue() << ", " << getDest() << ", " << getTileX() << ", "
          << getTileY() << ")";
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
  if (failed(verifyTileAttr(*this, getTileAttr(), tensorType)))
    return failure();
  return verifyMemRefAndTensor(*this, memRefType, tensorType);
}

LogicalResult StoreOp::verify() {
  auto tensorType = llvm::dyn_cast<RankedTensorType>(getValue().getType());
  auto memRefType = llvm::dyn_cast<MemRefType>(getDest().getType());
  if (!tensorType)
    return emitOpError("requires a ranked tensor operand");
  if (failed(verifyTileAttr(*this, getTileAttr(), tensorType)))
    return failure();
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
  if (lhsType.getRank() != 2 || !lhsType.hasStaticShape())
    return emitOpError("requires statically shaped rank-2 tensors");
  return success();
}

LogicalResult ExpOp::verify() {
  auto inputType = llvm::dyn_cast<RankedTensorType>(getInput().getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(getResult().getType());
  if (!inputType || !resultType)
    return emitOpError("requires ranked tensor operand and result");
  if (inputType != resultType)
    return emitOpError("requires operand and result tensor types to match");
  if (inputType.getRank() != 2 || !inputType.hasStaticShape())
    return emitOpError("requires statically shaped rank-2 tensors");
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
