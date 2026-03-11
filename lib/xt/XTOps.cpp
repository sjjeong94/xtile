#include "xt/XTOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::xt;

#define GET_OP_CLASSES
#include "xt/XTOps.cpp.inc"

static LogicalResult verifyMemRefAndTensor(Operation *op, MemRefType memRefType,
                                           RankedTensorType tensorType) {
  if (!memRefType)
    return op->emitOpError("requires a ranked memref operand");
  if (!tensorType || !tensorType.hasStaticShape())
    return op->emitOpError("requires a statically shaped tensor");
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

static ParseResult parseFunctionalTypeOp(
    OpAsmParser &parser, OperationState &result,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands, int32_t operandCount) {
  SmallVector<Type> operandTypes;
  Type resultType;
  if (parser.parseLParen() || parser.parseOperandList(operands, operandCount) ||
      parser.parseRParen() || parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColon() || parser.parseLParen() || parser.parseTypeList(operandTypes) ||
      parser.parseRParen() || parser.parseArrow() || parser.parseType(resultType))
    return failure();
  if (static_cast<int32_t>(operandTypes.size()) != operandCount)
    return parser.emitError(parser.getNameLoc(),
                            "operand count must match functional type");
  if (parser.resolveOperands(operands, operandTypes, parser.getNameLoc(),
                             result.operands))
    return failure();
  result.addTypes(resultType);
  return success();
}

static ParseResult parseSharedAttr(OpAsmParser &parser,
                                   OperationState &result) {
  Builder &builder = parser.getBuilder();
  if (failed(parser.parseOptionalLBrace()))
    return success();
  int64_t sharedValue;
  if (parser.parseKeyword("shared") || parser.parseEqual() ||
      parser.parseInteger(sharedValue) || parser.parseRBrace())
    return failure();
  result.addAttribute("shared", builder.getI64IntegerAttr(sharedValue));
  return success();
}

static void printSharedAttr(OpAsmPrinter &printer, IntegerAttr shared = {}) {
  if (!shared)
    return;
  printer << " {shared = " << shared.getInt() << "}";
}

static void printSingleTypeResultOp(OpAsmPrinter &printer, Operation *op) {
  printer << "(";
  printer.printOperands(op->getOperands());
  printer << ")";
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : " << op->getResult(0).getType();
}

static ParseResult parseConv2DAttrs(OpAsmParser &parser, OperationState &result) {
  Builder &builder = parser.getBuilder();
  SmallVector<int64_t> pad, stride, dilation;
  auto parseArray = [&](StringRef keyword,
                        SmallVectorImpl<int64_t> &values) -> ParseResult {
    if (parser.parseKeyword(keyword) || parser.parseEqual() ||
        parser.parseLSquare())
      return failure();
    do {
      int64_t value;
      if (parser.parseInteger(value))
        return failure();
      values.push_back(value);
    } while (succeeded(parser.parseOptionalComma()));
    return parser.parseRSquare();
  };

  if (parser.parseLBrace() || parseArray("pad", pad) || parser.parseComma() ||
      parseArray("stride", stride) || parser.parseComma() ||
      parseArray("dilation", dilation) || parser.parseRBrace())
    return failure();
  result.addAttribute("pad", builder.getDenseI64ArrayAttr(pad));
  result.addAttribute("stride", builder.getDenseI64ArrayAttr(stride));
  result.addAttribute("dilation", builder.getDenseI64ArrayAttr(dilation));
  return success();
}

static void printConv2DAttrs(OpAsmPrinter &printer, DenseI64ArrayAttr pad,
                             DenseI64ArrayAttr stride,
                             DenseI64ArrayAttr dilation) {
  auto printArray = [&](StringRef keyword, DenseI64ArrayAttr values) {
    printer << keyword << " = [";
    llvm::interleaveComma(values.asArrayRef(), printer,
                          [&](int64_t value) { printer << value; });
    printer << "]";
  };

  printer << " {";
  printArray("pad", pad);
  printer << ", ";
  printArray("stride", stride);
  printer << ", ";
  printArray("dilation", dilation);
  printer << "}";
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
      parseSharedAttr(parser, result) ||
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
  printSharedAttr(printer, getSharedAttr());
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
  printer << " : " << getValue().getType() << " -> " << getDest().getType();
}

ParseResult AddOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  return parseFunctionalTypeOp(parser, result, operands, 2);
}

void AddOp::print(OpAsmPrinter &printer) {
  printer << "(";
  printer.printOperands(getOperands());
  printer << ")";
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : (" << getLhs().getType() << ", " << getRhs().getType() << ") -> "
          << getResult().getType();
}

ParseResult SubOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  return parseFunctionalTypeOp(parser, result, operands, 2);
}

void SubOp::print(OpAsmPrinter &printer) {
  printer << "(";
  printer.printOperands(getOperands());
  printer << ")";
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : (" << getLhs().getType() << ", " << getRhs().getType() << ") -> "
          << getResult().getType();
}

ParseResult MulOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  return parseFunctionalTypeOp(parser, result, operands, 2);
}

void MulOp::print(OpAsmPrinter &printer) {
  printer << "(";
  printer.printOperands(getOperands());
  printer << ")";
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : (" << getLhs().getType() << ", " << getRhs().getType() << ") -> "
          << getResult().getType();
}

ParseResult ExpOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  return parseSingleTypeResultOp(parser, result, operands, 1);
}

void ExpOp::print(OpAsmPrinter &printer) { printSingleTypeResultOp(printer, *this); }

ParseResult CosOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  return parseSingleTypeResultOp(parser, result, operands, 1);
}

void CosOp::print(OpAsmPrinter &printer) { printSingleTypeResultOp(printer, *this); }

ParseResult SinOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  return parseSingleTypeResultOp(parser, result, operands, 1);
}

void SinOp::print(OpAsmPrinter &printer) { printSingleTypeResultOp(printer, *this); }

ParseResult ReciprocalOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  return parseSingleTypeResultOp(parser, result, operands, 1);
}

void ReciprocalOp::print(OpAsmPrinter &printer) {
  printSingleTypeResultOp(printer, *this);
}

ParseResult RsqrtOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  return parseSingleTypeResultOp(parser, result, operands, 1);
}

void RsqrtOp::print(OpAsmPrinter &printer) { printSingleTypeResultOp(printer, *this); }

ParseResult SigmoidOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  return parseSingleTypeResultOp(parser, result, operands, 1);
}

void SigmoidOp::print(OpAsmPrinter &printer) {
  printSingleTypeResultOp(printer, *this);
}

ParseResult TanhOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  return parseSingleTypeResultOp(parser, result, operands, 1);
}

void TanhOp::print(OpAsmPrinter &printer) { printSingleTypeResultOp(printer, *this); }

ParseResult SiluOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  return parseSingleTypeResultOp(parser, result, operands, 1);
}

void SiluOp::print(OpAsmPrinter &printer) { printSingleTypeResultOp(printer, *this); }

ParseResult MatmulOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  return parseFunctionalTypeOp(parser, result, operands, 2);
}

void MatmulOp::print(OpAsmPrinter &printer) {
  printer << "(";
  printer.printOperands(getOperands());
  printer << ")";
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : (" << getLhs().getType() << ", " << getRhs().getType() << ") -> "
          << getResult().getType();
}

ParseResult MMAOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  return parseFunctionalTypeOp(parser, result, operands, 3);
}

void MMAOp::print(OpAsmPrinter &printer) {
  printer << "(";
  printer.printOperands(getOperands());
  printer << ")";
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : (" << getLhs().getType() << ", " << getRhs().getType() << ", "
          << getAcc().getType() << ") -> " << getResult().getType();
}

ParseResult Conv2DOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<Type> operandTypes;
  Type resultType;
  if (parser.parseLParen() || parser.parseOperandList(operands, 2) ||
      parser.parseRParen() || parseConv2DAttrs(parser, result) ||
      parser.parseColon() || parser.parseLParen() ||
      parser.parseTypeList(operandTypes) || parser.parseRParen() ||
      parser.parseArrow() || parser.parseType(resultType))
    return failure();
  if (operandTypes.size() != 2)
    return parser.emitError(parser.getNameLoc(),
                            "conv2d expects exactly two operand types");
  if (parser.resolveOperands(operands, operandTypes, parser.getNameLoc(),
                             result.operands))
    return failure();
  result.addTypes(resultType);
  return success();
}

void Conv2DOp::print(OpAsmPrinter &printer) {
  printer << "(";
  printer.printOperands(getOperands());
  printer << ")";
  printConv2DAttrs(printer, getPadAttr(), getStrideAttr(), getDilationAttr());
  printer << " : (" << getInput().getType() << ", " << getFilter().getType()
          << ") -> " << getResult().getType();
}

LogicalResult LoadOp::verify() {
  auto tensorType = llvm::dyn_cast<RankedTensorType>(getResult().getType());
  auto memRefType = llvm::dyn_cast<MemRefType>(getSource().getType());
  if (!tensorType)
    return emitOpError("requires a ranked tensor result");
  if (getCoords().empty())
    return emitOpError("requires at least one coordinate");
  if (auto shared = getSharedAttr();
      shared && shared.getInt() != 0 && shared.getInt() != 1)
    return emitOpError("shared attribute must be 0 or 1");
  if (static_cast<int64_t>(getCoords().size()) != tensorType.getRank())
    return emitOpError("coordinate count must match tensor rank");
  return verifyMemRefAndTensor(*this, memRefType, tensorType);
}

LogicalResult StoreOp::verify() {
  auto tensorType = llvm::dyn_cast<RankedTensorType>(getValue().getType());
  auto memRefType = llvm::dyn_cast<MemRefType>(getDest().getType());
  if (!tensorType)
    return emitOpError("requires a ranked tensor operand");
  if (getCoords().empty())
    return emitOpError("requires at least one coordinate");
  if (static_cast<int64_t>(getCoords().size()) != tensorType.getRank())
    return emitOpError("coordinate count must match tensor rank");
  return verifyMemRefAndTensor(*this, memRefType, tensorType);
}

static FailureOr<SmallVector<int64_t>> computeBroadcastShape(RankedTensorType lhsType,
                                                             RankedTensorType rhsType) {
  if (!lhsType || !rhsType || !lhsType.hasStaticShape() || !rhsType.hasStaticShape())
    return failure();
  int64_t resultRank = std::max(lhsType.getRank(), rhsType.getRank());
  SmallVector<int64_t> shape(resultRank, 1);
  for (int64_t i = 0; i < resultRank; ++i) {
    int64_t lhsIndex = lhsType.getRank() - 1 - i;
    int64_t rhsIndex = rhsType.getRank() - 1 - i;
    int64_t lhsDim = lhsIndex >= 0 ? lhsType.getDimSize(lhsIndex) : 1;
    int64_t rhsDim = rhsIndex >= 0 ? rhsType.getDimSize(rhsIndex) : 1;
    if (lhsDim != rhsDim && lhsDim != 1 && rhsDim != 1)
      return failure();
    shape[resultRank - 1 - i] = std::max(lhsDim, rhsDim);
  }
  return shape;
}

static LogicalResult verifyBroadcastableTensorTypes(Operation *op, Value lhs, Value rhs,
                                                    Value result) {
  auto lhsType = llvm::dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = llvm::dyn_cast<RankedTensorType>(rhs.getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(result.getType());
  if (!lhsType || !rhsType || !resultType)
    return op->emitOpError("requires ranked tensor operands and result");
  if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() || !resultType.hasStaticShape())
    return op->emitOpError("requires statically shaped tensors");
  if (lhsType.getElementType() != rhsType.getElementType() ||
      lhsType.getElementType() != resultType.getElementType())
    return op->emitOpError("requires operand and result element types to match");
  FailureOr<SmallVector<int64_t>> broadcastShape = computeBroadcastShape(lhsType, rhsType);
  if (failed(broadcastShape))
    return op->emitOpError("operands are not broadcast-compatible with result tensor type");
  if (static_cast<int64_t>(broadcastShape->size()) != resultType.getRank())
    return op->emitOpError("operands are not broadcast-compatible with result tensor type");
  for (auto [expected, actual] : llvm::zip_equal(*broadcastShape, resultType.getShape())) {
    if (expected != actual)
      return op->emitOpError("operands are not broadcast-compatible with result tensor type");
  }
  return success();
}

static LogicalResult verifySameUnaryTensorTypes(Operation *op, Value input,
                                                Value result) {
  auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(result.getType());
  if (!inputType || !resultType)
    return op->emitOpError("requires ranked tensor operand and result");
  if (inputType != resultType)
    return op->emitOpError("requires operand and result tensor types to match");
  if (!inputType.hasStaticShape())
    return op->emitOpError("requires statically shaped tensors");
  return success();
}

LogicalResult AddOp::verify() {
  return verifyBroadcastableTensorTypes(*this, getLhs(), getRhs(), getResult());
}

LogicalResult SubOp::verify() {
  return verifyBroadcastableTensorTypes(*this, getLhs(), getRhs(), getResult());
}

LogicalResult MulOp::verify() {
  return verifyBroadcastableTensorTypes(*this, getLhs(), getRhs(), getResult());
}

LogicalResult ExpOp::verify() {
  return verifySameUnaryTensorTypes(*this, getInput(), getResult());
}

LogicalResult CosOp::verify() { return verifySameUnaryTensorTypes(*this, getInput(), getResult()); }
LogicalResult SinOp::verify() { return verifySameUnaryTensorTypes(*this, getInput(), getResult()); }
LogicalResult ReciprocalOp::verify() { return verifySameUnaryTensorTypes(*this, getInput(), getResult()); }
LogicalResult RsqrtOp::verify() { return verifySameUnaryTensorTypes(*this, getInput(), getResult()); }
LogicalResult SigmoidOp::verify() { return verifySameUnaryTensorTypes(*this, getInput(), getResult()); }
LogicalResult TanhOp::verify() { return verifySameUnaryTensorTypes(*this, getInput(), getResult()); }
LogicalResult SiluOp::verify() { return verifySameUnaryTensorTypes(*this, getInput(), getResult()); }

static LogicalResult verifyMatmulLikeShape(Operation *op, RankedTensorType lhsType,
                                           RankedTensorType rhsType,
                                           RankedTensorType resultType) {
  if (!lhsType || !rhsType || !resultType)
    return op->emitOpError("requires ranked tensor operands and result");
  if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
      !resultType.hasStaticShape())
    return op->emitOpError("requires statically shaped tensors");
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 || resultType.getRank() != 2)
    return op->emitOpError("requires rank-2 tensors");
  if (lhsType.getDimSize(1) != rhsType.getDimSize(0))
    return op->emitOpError(
        "matmul requires lhs inner dimension to match rhs outer dimension");
  if (resultType.getDimSize(0) != lhsType.getDimSize(0) ||
      resultType.getDimSize(1) != rhsType.getDimSize(1))
    return op->emitOpError("result shape must match matmul output shape");
  return success();
}

static FailureOr<int64_t> computeConvOutputDim(int64_t inputSize, int64_t kernelSize,
                                               int64_t padBefore, int64_t padAfter,
                                               int64_t stride, int64_t dilation) {
  int64_t effectiveKernel = dilation * (kernelSize - 1) + 1;
  int64_t numerator = inputSize + padBefore + padAfter - effectiveKernel;
  if (numerator < 0)
    return failure();
  return numerator / stride + 1;
}

LogicalResult MatmulOp::verify() {
  auto lhsType = llvm::dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsType = llvm::dyn_cast<RankedTensorType>(getRhs().getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(getResult().getType());
  if (failed(verifyMatmulLikeShape(*this, lhsType, rhsType, resultType)))
    return failure();
  if (lhsType.getElementType() == rhsType.getElementType() &&
      lhsType.getElementType() == resultType.getElementType())
    return success();
  if (lhsType.getElementType().isInteger(8) &&
      rhsType.getElementType().isInteger(8) &&
      (resultType.getElementType().isF32() || resultType.getElementType().isBF16()))
    return success();
  return emitOpError(
      "requires matching element types or i8 inputs with f32/bf16 result");
  return success();
}

LogicalResult MMAOp::verify() {
  auto lhsType = llvm::dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsType = llvm::dyn_cast<RankedTensorType>(getRhs().getType());
  auto accType = llvm::dyn_cast<RankedTensorType>(getAcc().getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(getResult().getType());
  if (failed(verifyMatmulLikeShape(*this, lhsType, rhsType, resultType)))
    return failure();
  if (!accType || !accType.hasStaticShape() || accType.getRank() != 2)
    return emitOpError("requires a rank-2 statically shaped accumulator");
  if (accType != resultType)
    return emitOpError("accumulator and result tensor types must match");
  if (!lhsType.getElementType().isInteger(8) || !rhsType.getElementType().isInteger(8))
    return emitOpError("mma requires i8 input tensors");
  Type accElem = accType.getElementType();
  if (!accElem.isF32() && !accElem.isBF16())
    return emitOpError("mma requires f32 or bf16 accumulator and result tensors");
  return success();
}

LogicalResult Conv2DOp::verify() {
  auto inputType = llvm::dyn_cast<RankedTensorType>(getInput().getType());
  auto filterType = llvm::dyn_cast<RankedTensorType>(getFilter().getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(getResult().getType());
  if (!inputType || !filterType || !resultType)
    return emitOpError("requires ranked tensor operands and result");
  if (!inputType.hasStaticShape() || !filterType.hasStaticShape() ||
      !resultType.hasStaticShape())
    return emitOpError("requires statically shaped tensors");
  if (inputType.getRank() != 4 || filterType.getRank() != 4 ||
      resultType.getRank() != 4)
    return emitOpError("requires rank-4 input, filter, and result tensors");
  if (getPadAttr().size() != 4)
    return emitOpError("pad attribute must have exactly 4 entries");
  if (getStrideAttr().size() != 2)
    return emitOpError("stride attribute must have exactly 2 entries");
  if (getDilationAttr().size() != 2)
    return emitOpError("dilation attribute must have exactly 2 entries");
  for (int64_t pad : getPadAttr().asArrayRef()) {
    if (pad < 0)
      return emitOpError("pad attribute entries must be non-negative");
  }
  for (int64_t stride : getStrideAttr().asArrayRef()) {
    if (stride <= 0)
      return emitOpError("stride attribute entries must be positive");
  }
  for (int64_t dilation : getDilationAttr().asArrayRef()) {
    if (dilation <= 0)
      return emitOpError("dilation attribute entries must be positive");
  }
  if (!inputType.getElementType().isInteger(8) ||
      !filterType.getElementType().isInteger(8))
    return emitOpError("conv2d requires i8 input and filter tensors");
  if (!resultType.getElementType().isF32() &&
      !resultType.getElementType().isBF16())
    return emitOpError("conv2d requires f32 or bf16 result tensors");
  if (inputType.getDimSize(3) != filterType.getDimSize(2))
    return emitOpError("conv2d requires input and filter channel dimensions to match");
  if (inputType.getDimSize(0) != resultType.getDimSize(0))
    return emitOpError("conv2d result batch dimension must match input");
  if (filterType.getDimSize(3) != resultType.getDimSize(3))
    return emitOpError("conv2d result channel dimension must match filter output channels");

  FailureOr<int64_t> outH = computeConvOutputDim(
      inputType.getDimSize(1), filterType.getDimSize(0), getPadAttr()[0], getPadAttr()[2],
      getStrideAttr()[0], getDilationAttr()[0]);
  FailureOr<int64_t> outW = computeConvOutputDim(
      inputType.getDimSize(2), filterType.getDimSize(1), getPadAttr()[1], getPadAttr()[3],
      getStrideAttr()[1], getDilationAttr()[1]);
  if (failed(outH) || failed(outW))
    return emitOpError("conv2d kernel configuration produces an invalid output shape");
  if (*outH != resultType.getDimSize(1) || *outW != resultType.getDimSize(2))
    return emitOpError("conv2d result spatial dimensions do not match pad/stride/dilation");
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

struct FoldSubZeroPattern : public OpRewritePattern<SubOp> {
  using OpRewritePattern<SubOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubOp op,
                                PatternRewriter &rewriter) const override {
    if (!isZeroTensor(op.getRhs()))
      return failure();
    rewriter.replaceOp(op, op.getLhs());
    return success();
  }
};
} // namespace

void AddOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<FoldAddZeroPattern>(context);
}

void SubOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<FoldSubZeroPattern>(context);
}
