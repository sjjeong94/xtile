from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Callable


@dataclass(frozen=True)
class DType:
    name: str
    mlir_name: str


float32 = DType(name="float32", mlir_name="f32")
int8 = DType(name="int8", mlir_name="i8")


@dataclass(frozen=True)
class Array:
    shape: tuple[int, ...]
    dtype: DType

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple) or not self.shape:
            raise TypeError("shape must be a non-empty tuple of integers")
        if any(not isinstance(dim, int) or dim <= 0 for dim in self.shape):
            raise TypeError("shape must contain positive integers")
        if not isinstance(self.dtype, DType):
            raise TypeError("dtype must be an xtile dtype")

    @property
    def ndim(self) -> int:
        return len(self.shape)


class KernelFunction:
    def __init__(self, fn: Callable[..., object]) -> None:
        self.fn = fn
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.__wrapped__ = fn


@dataclass(frozen=True)
class KernelArg:
    index: int
    python_name: str
    array: Array


@dataclass(frozen=True)
class ScalarArg:
    index: int
    python_name: str
    value: int


@dataclass(frozen=True)
class BlockId:
    dim: int


@dataclass(frozen=True)
class ConstantIndex:
    value: int


@dataclass(frozen=True)
class TensorValue:
    ssa_name: str
    shape: tuple[int, ...]
    dtype: DType

    def __sub__(self, other: object) -> TensorValue:
        return sub(self, other)

    def __mul__(self, other: object) -> TensorValue:
        return mul(self, other)

    def __add__(self, other: object) -> TensorValue:
        return add(self, other)


@dataclass(frozen=True)
class LoadOp:
    result: TensorValue
    array: KernelArg
    index: tuple[BlockId | ConstantIndex, ...]
    shared: int | None


@dataclass(frozen=True)
class StoreOp:
    array: KernelArg
    index: tuple[BlockId | ConstantIndex, ...]
    tile: TensorValue


@dataclass(frozen=True)
class ReduceOp:
    op_name: str
    result: TensorValue
    input_value: TensorValue


@dataclass(frozen=True)
class UnaryOp:
    op_name: str
    result: TensorValue
    input_value: TensorValue


@dataclass(frozen=True)
class BinaryOp:
    op_name: str
    result: TensorValue
    lhs: TensorValue
    rhs: TensorValue


@dataclass(frozen=True)
class CastOp:
    op_name: str
    result: TensorValue
    input_value: TensorValue


@dataclass(frozen=True)
class TernaryOp:
    op_name: str
    result: TensorValue
    lhs: TensorValue
    rhs: TensorValue
    third: TensorValue


@dataclass(frozen=True)
class AttrOp:
    op_name: str
    result: TensorValue
    operands: tuple[TensorValue, ...]
    attrs: tuple[tuple[str, tuple[int, ...]], ...]


class TraceContext:
    def __init__(
        self,
        kernel: KernelFunction,
        args: tuple[Array | int, ...],
        grid: tuple[int, int, int],
        double_buffering: bool,
    ) -> None:
        if len(grid) != 3 or any(not isinstance(dim, int) or dim <= 0 for dim in grid):
            raise ValueError("grid must be a 3-tuple of positive integers")

        signature = inspect.signature(kernel.fn)
        params = list(signature.parameters.values())
        if len(params) != len(args):
            raise ValueError("convert args must match the kernel signature")

        self.kernel_name = kernel.fn.__name__
        self.grid = grid
        self.double_buffering = double_buffering
        kernel_args: list[KernelArg | ScalarArg] = []
        for i, (param, arg) in enumerate(zip(params, args)):
            if isinstance(arg, Array):
                kernel_args.append(
                    KernelArg(index=i, python_name=param.name, array=arg)
                )
                continue
            if isinstance(arg, int):
                kernel_args.append(
                    ScalarArg(index=i, python_name=param.name, value=arg)
                )
                continue
            raise TypeError("convert args must be xt.Array values or integers")
        self.kernel_args = tuple(kernel_args)
        self.array_args = tuple(
            arg for arg in self.kernel_args if isinstance(arg, KernelArg)
        )
        self.operations: list[
            LoadOp
            | StoreOp
            | ReduceOp
            | UnaryOp
            | BinaryOp
            | CastOp
            | TernaryOp
            | AttrOp
        ] = []
        self.used_block_ids = False
        self.load_counts = [0] * len(self.kernel_args)
        self.store_counts = [0] * len(self.kernel_args)
        self.name_counts: dict[str, int] = {}
        self.constant_names: dict[int, str] = {}

    def trace_arguments(self) -> tuple[KernelArg | int, ...]:
        return tuple(
            arg if isinstance(arg, KernelArg) else arg.value for arg in self.kernel_args
        )

    def bid(self, dim: int) -> BlockId:
        if dim not in (0, 1, 2):
            raise ValueError("xt.bid only supports dimensions 0, 1, and 2")
        self.used_block_ids = True
        return BlockId(dim=dim)

    def load(
        self,
        array: KernelArg,
        index: tuple[BlockId | int, ...],
        shape: tuple[int, ...],
        shared: int | None = None,
    ) -> TensorValue:
        self._validate_kernel_arg(array)
        normalized_index = self._normalize_index(index)
        self._validate_shape(shape)
        self._validate_shared(shared)
        if len(array.array.shape) != len(shape):
            raise ValueError(
                "xt.load requires shape rank to match the source array rank"
            )
        if len(normalized_index) != len(shape):
            raise ValueError(
                "xt.load requires index rank to match the source array rank"
            )
        result = TensorValue(
            ssa_name=self._next_name("x"),
            shape=shape,
            dtype=array.array.dtype,
        )
        self.operations.append(
            LoadOp(result=result, array=array, index=normalized_index, shared=shared)
        )
        self.load_counts[array.index] += 1
        return result

    def reduce_max(self, input_value: TensorValue) -> TensorValue:
        return self._reduce("xt.reduce_max", input_value, "row_max")

    def reduce_sum(self, input_value: TensorValue) -> TensorValue:
        return self._reduce("xt.reduce_sum", input_value, "row_sum")

    def sub(self, lhs: TensorValue, rhs: TensorValue) -> TensorValue:
        return self._binary("xt.sub", lhs, rhs, "shifted")

    def add(self, lhs: TensorValue, rhs: TensorValue) -> TensorValue:
        return self._binary("xt.add", lhs, rhs, "sum")

    def exp(self, input_value: TensorValue) -> TensorValue:
        return self._unary("xt.exp", input_value, "exp")

    def cos(self, input_value: TensorValue) -> TensorValue:
        return self._unary("xt.cos", input_value, "cos")

    def sin(self, input_value: TensorValue) -> TensorValue:
        return self._unary("xt.sin", input_value, "sin")

    def reciprocal(self, input_value: TensorValue) -> TensorValue:
        return self._unary("xt.reciprocal", input_value, "inv_sum")

    def rsqrt(self, input_value: TensorValue) -> TensorValue:
        return self._unary("xt.rsqrt", input_value, "rsqrt")

    def sigmoid(self, input_value: TensorValue) -> TensorValue:
        return self._unary("xt.sigmoid", input_value, "sigmoid")

    def tanh(self, input_value: TensorValue) -> TensorValue:
        return self._unary("xt.tanh", input_value, "tanh")

    def silu(self, input_value: TensorValue) -> TensorValue:
        return self._unary("xt.silu", input_value, "silu")

    def mul(self, lhs: TensorValue, rhs: TensorValue) -> TensorValue:
        return self._binary("xt.mul", lhs, rhs, "normalized")

    def matmul(self, lhs: TensorValue, rhs: TensorValue) -> TensorValue:
        self._validate_tensor_value(lhs)
        self._validate_tensor_value(rhs)
        if len(lhs.shape) != 2 or len(rhs.shape) != 2:
            raise ValueError("xt.matmul currently supports rank-2 tensors only")
        if lhs.shape[1] != rhs.shape[0]:
            raise ValueError(
                "xt.matmul requires lhs inner dimension to match rhs outer dimension"
            )

        if lhs.dtype == int8 and rhs.dtype == int8:
            result_dtype = float32
        elif lhs.dtype == rhs.dtype:
            result_dtype = lhs.dtype
        else:
            raise TypeError("xt.matmul requires matching dtypes or int8 inputs")

        result = TensorValue(
            ssa_name=self._next_name("matmul"),
            shape=(lhs.shape[0], rhs.shape[1]),
            dtype=result_dtype,
        )
        self.operations.append(
            BinaryOp(op_name="xt.matmul", result=result, lhs=lhs, rhs=rhs)
        )
        return result

    def astype(self, input_value: TensorValue, dtype: DType) -> TensorValue:
        self._validate_tensor_value(input_value)
        if not isinstance(dtype, DType):
            raise TypeError("xt.astype requires an xtile dtype")
        if input_value.dtype == dtype:
            return input_value

        if input_value.dtype == int8 and dtype == float32:
            op_name = "xt.itof"
        elif input_value.dtype == float32 and dtype == int8:
            op_name = "xt.ftoi"
        else:
            raise TypeError("xt.astype only supports int8<->float32 conversions")

        result = TensorValue(
            ssa_name=self._next_name("cast"),
            shape=input_value.shape,
            dtype=dtype,
        )
        self.operations.append(
            CastOp(op_name=op_name, result=result, input_value=input_value)
        )
        return result

    def reshape(self, input_value: TensorValue, shape: tuple[int, ...]) -> TensorValue:
        self._validate_tensor_value(input_value)
        self._validate_shape(shape)
        if self._element_count(input_value.shape) != self._element_count(shape):
            raise ValueError("xt.reshape requires the same number of elements")

        result = TensorValue(
            ssa_name=self._next_name("reshape"),
            shape=shape,
            dtype=input_value.dtype,
        )
        self.operations.append(
            UnaryOp(op_name="xt.reshape", result=result, input_value=input_value)
        )
        return result

    def transpose(self, input_value: TensorValue) -> TensorValue:
        self._validate_tensor_value(input_value)
        if len(input_value.shape) != 3:
            raise ValueError("xt.transpose currently supports rank-3 tensors only")
        result = TensorValue(
            ssa_name=self._next_name("transpose"),
            shape=(input_value.shape[0], input_value.shape[2], input_value.shape[1]),
            dtype=input_value.dtype,
        )
        self.operations.append(
            UnaryOp(op_name="xt.transpose", result=result, input_value=input_value)
        )
        return result

    def mma(self, lhs: TensorValue, rhs: TensorValue, acc: TensorValue) -> TensorValue:
        self._validate_tensor_value(lhs)
        self._validate_tensor_value(rhs)
        self._validate_tensor_value(acc)
        if len(lhs.shape) != 2 or len(rhs.shape) != 2 or len(acc.shape) != 2:
            raise ValueError("xt.mma currently supports rank-2 tensors only")
        if lhs.dtype != int8 or rhs.dtype != int8:
            raise TypeError("xt.mma requires int8 lhs and rhs tensors")
        if acc.dtype != float32:
            raise TypeError("xt.mma currently requires a float32 accumulator")
        if lhs.shape[1] != rhs.shape[0]:
            raise ValueError(
                "xt.mma requires lhs inner dimension to match rhs outer dimension"
            )
        if acc.shape != (lhs.shape[0], rhs.shape[1]):
            raise ValueError(
                "xt.mma accumulator shape must match the matmul result shape"
            )

        result = TensorValue(
            ssa_name=self._next_name("mma"),
            shape=acc.shape,
            dtype=acc.dtype,
        )
        self.operations.append(
            TernaryOp(op_name="xt.mma", result=result, lhs=lhs, rhs=rhs, third=acc)
        )
        return result

    def conv2d(
        self,
        input_value: TensorValue,
        filter_value: TensorValue,
        *,
        pad: tuple[int, int, int, int],
        stride: tuple[int, int],
        dilation: tuple[int, int],
    ) -> TensorValue:
        return self._conv_like(
            "xt.conv2d",
            input_value,
            filter_value,
            pad=pad,
            stride=stride,
            dilation=dilation,
            depthwise=False,
        )

    def depthwise_conv2d(
        self,
        input_value: TensorValue,
        filter_value: TensorValue,
        *,
        pad: tuple[int, int, int, int],
        stride: tuple[int, int],
        dilation: tuple[int, int],
    ) -> TensorValue:
        return self._conv_like(
            "xt.depthwise_conv2d",
            input_value,
            filter_value,
            pad=pad,
            stride=stride,
            dilation=dilation,
            depthwise=True,
        )

    def store(
        self,
        array: KernelArg,
        index: tuple[BlockId | int, ...],
        tile: TensorValue,
    ) -> None:
        self._validate_kernel_arg(array)
        normalized_index = self._normalize_index(index)
        if not isinstance(tile, TensorValue):
            raise TypeError("xt.store expects a tensor value")
        if len(array.array.shape) != len(tile.shape):
            raise ValueError(
                "xt.store requires tile rank to match the destination array rank"
            )
        if len(normalized_index) != len(tile.shape):
            raise ValueError(
                "xt.store requires index rank to match the destination array rank"
            )
        self.operations.append(StoreOp(array=array, index=normalized_index, tile=tile))
        self.store_counts[array.index] += 1

    def emit_function_asm(self) -> str:
        arg_names = self._assign_mlir_arg_names()
        arg_defs = ", ".join(
            f"%{arg_names[arg.index]}: {self._format_memref_type(arg.array)}"
            for arg in self.array_args
        )

        attributes = [
            f"xt.grid = array<i32: {', '.join(str(dim) for dim in self.grid)}>"
        ]
        if self.double_buffering:
            attributes.append("xt.double_buffering = 1 : i32")

        lines = [
            f"func.func @{self.kernel_name}({arg_defs}) attributes {{{', '.join(attributes)}}} {{"
        ]
        if self.used_block_ids:
            lines.append("  %bid:3 = xt.get_tile_block_id : i32, i32, i32")

        if self.constant_names:
            for value, name in sorted(self.constant_names.items()):
                lines.append(f"  {name} = arith.constant {value} : i32")
            lines.append("")

        for op in self.operations:
            if isinstance(op, LoadOp):
                attr = " {shared = 1 : i64}" if op.shared == 1 else ""
                if op.shared == 2:
                    attr = " {shared = 2 : i64}"
                lines.append(
                    "  "
                    f"{op.result.ssa_name} = xt.load(%{arg_names[op.array.index]}, "
                    f"{self._format_index(op.index)}){attr} : "
                    f"({self._format_operand_type_list((self._format_memref_type(op.array.array),), len(op.index))}) -> "
                    f"{self._format_tensor_type(op.result.shape, op.result.dtype)}"
                )
                continue

            if isinstance(op, StoreOp):
                lines.append(
                    "  "
                    f"xt.store({op.tile.ssa_name}, %{arg_names[op.array.index]}, {self._format_index(op.index)}) : "
                    f"({self._format_operand_type_list((self._format_tensor_type(op.tile.shape, op.tile.dtype), self._format_memref_type(op.array.array)), len(op.index))}) -> ()"
                )
                continue

            if isinstance(op, ReduceOp):
                lines.append(
                    "  "
                    f"{op.result.ssa_name} = {op.op_name}({op.input_value.ssa_name}) : "
                    f"{self._format_tensor_type(op.input_value.shape, op.input_value.dtype)} -> "
                    f"{self._format_tensor_type(op.result.shape, op.result.dtype)}"
                )
                continue

            if isinstance(op, UnaryOp):
                lines.append(
                    "  "
                    f"{op.result.ssa_name} = {op.op_name}({op.input_value.ssa_name}) : "
                    f"{self._format_tensor_type(op.input_value.shape, op.input_value.dtype)} -> "
                    f"{self._format_tensor_type(op.result.shape, op.result.dtype)}"
                )
                continue

            if isinstance(op, CastOp):
                lines.append(
                    "  "
                    f"{op.result.ssa_name} = {op.op_name}({op.input_value.ssa_name}) : "
                    f"{self._format_tensor_type(op.input_value.shape, op.input_value.dtype)} -> "
                    f"{self._format_tensor_type(op.result.shape, op.result.dtype)}"
                )
                continue

            if isinstance(op, TernaryOp):
                lines.append(
                    "  "
                    f"{op.result.ssa_name} = {op.op_name}({op.lhs.ssa_name}, {op.rhs.ssa_name}, {op.third.ssa_name}) : "
                    f"{self._format_tensor_type(op.lhs.shape, op.lhs.dtype)}, "
                    f"{self._format_tensor_type(op.rhs.shape, op.rhs.dtype)}, "
                    f"{self._format_tensor_type(op.third.shape, op.third.dtype)} -> "
                    f"{self._format_tensor_type(op.result.shape, op.result.dtype)}"
                )
                continue

            if isinstance(op, AttrOp):
                attr_text = ", ".join(
                    f"{name} = array<i64: {', '.join(str(v) for v in values)}>"
                    for name, values in op.attrs
                )
                operand_types = ", ".join(
                    self._format_tensor_type(operand.shape, operand.dtype)
                    for operand in op.operands
                )
                lines.append(
                    "  "
                    f"{op.result.ssa_name} = {op.op_name}("
                    + ", ".join(operand.ssa_name for operand in op.operands)
                    + f") {{{attr_text}}} : {operand_types} -> "
                    f"{self._format_tensor_type(op.result.shape, op.result.dtype)}"
                )
                continue

            lines.append(
                "  "
                f"{op.result.ssa_name} = {op.op_name}({op.lhs.ssa_name}, {op.rhs.ssa_name}) : "
                f"{self._format_tensor_type(op.lhs.shape, op.lhs.dtype)}, "
                f"{self._format_tensor_type(op.rhs.shape, op.rhs.dtype)} -> "
                f"{self._format_tensor_type(op.result.shape, op.result.dtype)}"
            )

        lines.append("  func.return")
        lines.append("}")
        return "\n".join(lines)

    def _reduce(self, op_name: str, input_value: TensorValue, stem: str) -> TensorValue:
        self._validate_tensor_value(input_value)
        if len(input_value.shape) != 2:
            raise ValueError("xt.max and xt.sum currently support rank-2 tensors only")
        result = TensorValue(
            ssa_name=self._next_name(stem),
            shape=(input_value.shape[0], 1),
            dtype=input_value.dtype,
        )
        self.operations.append(
            ReduceOp(op_name=op_name, result=result, input_value=input_value)
        )
        return result

    def _unary(self, op_name: str, input_value: TensorValue, stem: str) -> TensorValue:
        self._validate_tensor_value(input_value)
        result = TensorValue(
            ssa_name=self._next_name(stem),
            shape=input_value.shape,
            dtype=input_value.dtype,
        )
        self.operations.append(
            UnaryOp(op_name=op_name, result=result, input_value=input_value)
        )
        return result

    def _binary(
        self, op_name: str, lhs: TensorValue, rhs: TensorValue, stem: str
    ) -> TensorValue:
        self._validate_tensor_value(lhs)
        self._validate_tensor_value(rhs)
        if lhs.dtype != rhs.dtype:
            raise TypeError("binary xtile ops require matching dtypes")
        if not self._is_broadcast_compatible(lhs.shape, rhs.shape):
            raise ValueError(
                "binary xtile ops require equal or rowwise-broadcast-compatible shapes"
            )

        result_shape = lhs.shape
        result = TensorValue(
            ssa_name=self._next_name(stem),
            shape=result_shape,
            dtype=lhs.dtype,
        )
        self.operations.append(
            BinaryOp(op_name=op_name, result=result, lhs=lhs, rhs=rhs)
        )
        return result

    def _normalize_index(
        self,
        index: tuple[BlockId | int, ...],
    ) -> tuple[BlockId | ConstantIndex, ...]:
        if not isinstance(index, tuple) or not index:
            raise TypeError("index must be a non-empty tuple of values")

        normalized: list[BlockId | ConstantIndex] = []
        for value in index:
            if isinstance(value, BlockId):
                normalized.append(value)
                continue
            if isinstance(value, int):
                normalized.append(self._constant_index(value))
                continue
            raise TypeError("index values must be xt.bid(...) results or integers")
        return tuple(normalized)

    def _constant_index(self, value: int) -> ConstantIndex:
        if value not in self.constant_names:
            self.constant_names[value] = (
                "%zero" if value == 0 else self._next_name(f"c{value}")
            )
        return ConstantIndex(value=value)

    def _assign_mlir_arg_names(self) -> dict[int, str]:
        names: dict[int, str] = {}
        input_count = 0
        output_count = 0
        for arg in self.kernel_args:
            if not isinstance(arg, KernelArg):
                continue
            loads = self.load_counts[arg.index]
            stores = self.store_counts[arg.index]
            if loads > 0 and stores == 0:
                input_count += 1
                names[arg.index] = (
                    "input" if input_count == 1 else f"input{input_count}"
                )
            elif stores > 0 and loads == 0:
                output_count += 1
                names[arg.index] = (
                    "output" if output_count == 1 else f"output{output_count}"
                )
            else:
                names[arg.index] = arg.python_name
        return names

    def _next_name(self, stem: str) -> str:
        count = self.name_counts.get(stem, 0) + 1
        self.name_counts[stem] = count
        return f"%{stem}" if count == 1 else f"%{stem}{count}"

    @staticmethod
    def _validate_kernel_arg(array: KernelArg) -> None:
        if not isinstance(array, KernelArg):
            raise TypeError("xtile load/store expect kernel arguments")

    @staticmethod
    def _validate_tensor_value(value: TensorValue) -> None:
        if not isinstance(value, TensorValue):
            raise TypeError("xtile operation expects a tensor value")

    @staticmethod
    def _validate_shape(shape: tuple[int, ...]) -> None:
        if (
            not isinstance(shape, tuple)
            or not shape
            or any(not isinstance(dim, int) or dim <= 0 for dim in shape)
        ):
            raise TypeError("shape must be a tuple of positive integers")

    @staticmethod
    def _is_broadcast_compatible(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> bool:
        if lhs == rhs:
            return True
        if len(lhs) != len(rhs):
            return False
        return all(l_dim == r_dim or r_dim == 1 for l_dim, r_dim in zip(lhs, rhs))

    @staticmethod
    def _validate_shared(shared: int | None) -> None:
        if shared is not None and shared not in (0, 1, 2):
            raise ValueError("shared must be 0, 1, 2, or None")

    @staticmethod
    def _format_index(
        index: tuple[BlockId | ConstantIndex, BlockId | ConstantIndex],
    ) -> str:
        parts = []
        for value in index:
            if isinstance(value, BlockId):
                parts.append(f"%bid#{value.dim}")
            else:
                parts.append("%zero" if value.value == 0 else f"%c{value.value}")
        return ", ".join(parts)

    @staticmethod
    def _format_operand_type_list(
        prefix_types: tuple[str, ...], coord_count: int
    ) -> str:
        return ", ".join((*prefix_types, *(["i32"] * coord_count)))

    @staticmethod
    def _element_count(shape: tuple[int, ...]) -> int:
        count = 1
        for dim in shape:
            count *= dim
        return count

    @staticmethod
    def _validate_attr_tuple(
        name: str, value: tuple[int, ...], expected_len: int, *, positive: bool = False
    ) -> None:
        if not isinstance(value, tuple) or len(value) != expected_len:
            raise TypeError(f"{name} must be a tuple of {expected_len} integers")
        for item in value:
            if not isinstance(item, int):
                raise TypeError(f"{name} must be a tuple of {expected_len} integers")
            if positive and item <= 0:
                raise ValueError(f"{name} values must be positive")
            if not positive and item < 0:
                raise ValueError(f"{name} values must be non-negative")

    def _conv_like(
        self,
        op_name: str,
        input_value: TensorValue,
        filter_value: TensorValue,
        *,
        pad: tuple[int, int, int, int],
        stride: tuple[int, int],
        dilation: tuple[int, int],
        depthwise: bool,
    ) -> TensorValue:
        self._validate_tensor_value(input_value)
        self._validate_tensor_value(filter_value)
        self._validate_attr_tuple("pad", pad, 4)
        self._validate_attr_tuple("stride", stride, 2, positive=True)
        self._validate_attr_tuple("dilation", dilation, 2, positive=True)
        if len(input_value.shape) != 4 or len(filter_value.shape) != 4:
            raise ValueError(f"{op_name} currently supports rank-4 tensors only")
        if input_value.dtype != int8 or filter_value.dtype != int8:
            raise TypeError(f"{op_name} currently requires int8 inputs")
        if depthwise:
            if filter_value.shape[2] != 1:
                raise ValueError("xt.depthwise_conv2d requires filter dim 2 to be 1")
            if filter_value.shape[3] != input_value.shape[3]:
                raise ValueError("xt.depthwise_conv2d requires matching channel counts")
            out_channels = input_value.shape[3]
        else:
            if input_value.shape[3] != filter_value.shape[2]:
                raise ValueError("xt.conv2d requires input/filter channels to match")
            out_channels = filter_value.shape[3]

        out_h = self._compute_conv_output_dim(
            input_value.shape[1],
            filter_value.shape[0],
            pad[0],
            pad[2],
            stride[0],
            dilation[0],
        )
        out_w = self._compute_conv_output_dim(
            input_value.shape[2],
            filter_value.shape[1],
            pad[1],
            pad[3],
            stride[1],
            dilation[1],
        )
        result = TensorValue(
            ssa_name=self._next_name("conv" if not depthwise else "dwconv"),
            shape=(input_value.shape[0], out_h, out_w, out_channels),
            dtype=float32,
        )
        self.operations.append(
            AttrOp(
                op_name=op_name,
                result=result,
                operands=(input_value, filter_value),
                attrs=(("pad", pad), ("stride", stride), ("dilation", dilation)),
            )
        )
        return result

    @staticmethod
    def _compute_conv_output_dim(
        input_size: int,
        kernel_size: int,
        pad_before: int,
        pad_after: int,
        stride: int,
        dilation: int,
    ) -> int:
        effective_kernel = dilation * (kernel_size - 1) + 1
        numerator = input_size + pad_before + pad_after - effective_kernel
        if numerator < 0:
            raise ValueError("invalid convolution configuration")
        return numerator // stride + 1

    @staticmethod
    def _format_memref_type(array: Array) -> str:
        dims = "x".join(str(dim) for dim in array.shape)
        return f"memref<{dims}x{array.dtype.mlir_name}>"

    @staticmethod
    def _format_tensor_type(shape: tuple[int, ...], dtype: DType) -> str:
        dims = "x".join(str(dim) for dim in shape)
        return f"tensor<{dims}x{dtype.mlir_name}>"


_ACTIVE_TRACE: TraceContext | None = None


def kernel(fn: Callable[..., object]) -> KernelFunction:
    return KernelFunction(fn)


def bid(dim: int) -> BlockId:
    if _ACTIVE_TRACE is None:
        raise RuntimeError("xt.bid may only be used while converting a kernel")
    return _ACTIVE_TRACE.bid(dim)


def load(
    array: KernelArg,
    *,
    index: tuple[BlockId | int, ...],
    shape: tuple[int, ...],
    shared: int | None = None,
) -> TensorValue:
    if _ACTIVE_TRACE is None:
        raise RuntimeError("xt.load may only be used while converting a kernel")
    return _ACTIVE_TRACE.load(array, index, shape, shared)


def max(input_value: TensorValue) -> TensorValue:
    if _ACTIVE_TRACE is None:
        raise RuntimeError("xt.max may only be used while converting a kernel")
    return _ACTIVE_TRACE.reduce_max(input_value)


def sum(input_value: TensorValue) -> TensorValue:
    if _ACTIVE_TRACE is None:
        raise RuntimeError("xt.sum may only be used while converting a kernel")
    return _ACTIVE_TRACE.reduce_sum(input_value)


def sub(lhs: TensorValue, rhs: TensorValue) -> TensorValue:
    if _ACTIVE_TRACE is None:
        raise RuntimeError("xt.sub may only be used while converting a kernel")
    return _ACTIVE_TRACE.sub(lhs, rhs)


def exp(input_value: TensorValue) -> TensorValue:
    if _ACTIVE_TRACE is None:
        raise RuntimeError("xt.exp may only be used while converting a kernel")
    return _ACTIVE_TRACE.exp(input_value)


def cos(input_value: TensorValue) -> TensorValue:
    if _ACTIVE_TRACE is None:
        raise RuntimeError("xt.cos may only be used while converting a kernel")
    return _ACTIVE_TRACE.cos(input_value)


def sin(input_value: TensorValue) -> TensorValue:
    if _ACTIVE_TRACE is None:
        raise RuntimeError("xt.sin may only be used while converting a kernel")
    return _ACTIVE_TRACE.sin(input_value)


def reciprocal(input_value: TensorValue) -> TensorValue:
    if _ACTIVE_TRACE is None:
        raise RuntimeError("xt.reciprocal may only be used while converting a kernel")
    return _ACTIVE_TRACE.reciprocal(input_value)


def rsqrt(input_value: TensorValue) -> TensorValue:
    if _ACTIVE_TRACE is None:
        raise RuntimeError("xt.rsqrt may only be used while converting a kernel")
    return _ACTIVE_TRACE.rsqrt(input_value)


def sigmoid(input_value: TensorValue) -> TensorValue:
    if _ACTIVE_TRACE is None:
        raise RuntimeError("xt.sigmoid may only be used while converting a kernel")
    return _ACTIVE_TRACE.sigmoid(input_value)


def tanh(input_value: TensorValue) -> TensorValue:
    if _ACTIVE_TRACE is None:
        raise RuntimeError("xt.tanh may only be used while converting a kernel")
    return _ACTIVE_TRACE.tanh(input_value)


def silu(input_value: TensorValue) -> TensorValue:
    if _ACTIVE_TRACE is None:
        raise RuntimeError("xt.silu may only be used while converting a kernel")
    return _ACTIVE_TRACE.silu(input_value)


def mul(lhs: TensorValue, rhs: TensorValue) -> TensorValue:
    if _ACTIVE_TRACE is None:
        raise RuntimeError("xt.mul may only be used while converting a kernel")
    return _ACTIVE_TRACE.mul(lhs, rhs)


def add(lhs: TensorValue, rhs: TensorValue) -> TensorValue:
    if _ACTIVE_TRACE is None:
        raise RuntimeError("xt.add may only be used while converting a kernel")
    return _ACTIVE_TRACE.add(lhs, rhs)


def matmul(lhs: TensorValue, rhs: TensorValue) -> TensorValue:
    if _ACTIVE_TRACE is None:
        raise RuntimeError("xt.matmul may only be used while converting a kernel")
    return _ACTIVE_TRACE.matmul(lhs, rhs)


def astype(input_value: TensorValue, *, dtype: DType) -> TensorValue:
    if _ACTIVE_TRACE is None:
        raise RuntimeError("xt.astype may only be used while converting a kernel")
    return _ACTIVE_TRACE.astype(input_value, dtype)


def reshape(input_value: TensorValue, *, shape: tuple[int, ...]) -> TensorValue:
    if _ACTIVE_TRACE is None:
        raise RuntimeError("xt.reshape may only be used while converting a kernel")
    return _ACTIVE_TRACE.reshape(input_value, shape)


def transpose(input_value: TensorValue) -> TensorValue:
    if _ACTIVE_TRACE is None:
        raise RuntimeError("xt.transpose may only be used while converting a kernel")
    return _ACTIVE_TRACE.transpose(input_value)


def mma(lhs: TensorValue, rhs: TensorValue, acc: TensorValue) -> TensorValue:
    if _ACTIVE_TRACE is None:
        raise RuntimeError("xt.mma may only be used while converting a kernel")
    return _ACTIVE_TRACE.mma(lhs, rhs, acc)


def conv2d(
    input_value: TensorValue,
    filter_value: TensorValue,
    *,
    pad: tuple[int, int, int, int],
    stride: tuple[int, int],
    dilation: tuple[int, int],
) -> TensorValue:
    if _ACTIVE_TRACE is None:
        raise RuntimeError("xt.conv2d may only be used while converting a kernel")
    return _ACTIVE_TRACE.conv2d(
        input_value, filter_value, pad=pad, stride=stride, dilation=dilation
    )


def depthwise_conv2d(
    input_value: TensorValue,
    filter_value: TensorValue,
    *,
    pad: tuple[int, int, int, int],
    stride: tuple[int, int],
    dilation: tuple[int, int],
) -> TensorValue:
    if _ACTIVE_TRACE is None:
        raise RuntimeError(
            "xt.depthwise_conv2d may only be used while converting a kernel"
        )
    return _ACTIVE_TRACE.depthwise_conv2d(
        input_value, filter_value, pad=pad, stride=stride, dilation=dilation
    )


def store(
    array: KernelArg,
    *,
    index: tuple[BlockId | int, ...],
    tile: TensorValue,
) -> None:
    if _ACTIVE_TRACE is None:
        raise RuntimeError("xt.store may only be used while converting a kernel")
    _ACTIVE_TRACE.store(array, index, tile)


def convert(
    kernel_fn: KernelFunction,
    *,
    args: tuple[Array | int, ...],
    grid: tuple[int, int, int],
    double_buffering: bool,
    parse_module: Callable[[str], object],
) -> object:
    if not isinstance(kernel_fn, KernelFunction):
        raise TypeError("xt.convert expects a function decorated with @xt.kernel")

    global _ACTIVE_TRACE
    trace = TraceContext(
        kernel=kernel_fn,
        args=args,
        grid=grid,
        double_buffering=double_buffering,
    )
    previous_trace = _ACTIVE_TRACE
    _ACTIVE_TRACE = trace
    try:
        result = kernel_fn.fn(*trace.trace_arguments())
    finally:
        _ACTIVE_TRACE = previous_trace

    if result is not None:
        raise ValueError("xt.kernel functions must not return a value")

    return parse_module(trace.emit_function_asm())
