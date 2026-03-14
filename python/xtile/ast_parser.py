from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
import inspect
import textwrap

from .errors import XTConversionError
from .types import MemRefSpec, TensorSpec


@dataclass(frozen=True)
class IntExpr:
    kind: str
    value: int
    name: str | None = None


@dataclass(frozen=True)
class ConstOp:
    name: str
    value: int


@dataclass(frozen=True)
class LoadOp:
    name: str
    source: str
    indices: tuple[IntExpr, ...]
    shape: tuple[int, ...]
    shared: int | None
    result: TensorSpec


@dataclass(frozen=True)
class BinaryOp:
    name: str
    op_name: str
    lhs: str
    rhs: str
    result: TensorSpec


@dataclass(frozen=True)
class MatmulOp:
    name: str
    lhs: str
    rhs: str
    result: TensorSpec


@dataclass(frozen=True)
class UnaryOp:
    name: str
    op_name: str
    operand: str
    result: TensorSpec


@dataclass(frozen=True)
class FullOp:
    name: str
    shape: tuple[int, ...]
    value: float
    result: TensorSpec


@dataclass(frozen=True)
class ReshapeOp:
    name: str
    operand: str
    shape: tuple[int, ...]
    result: TensorSpec


@dataclass(frozen=True)
class TransposeOp:
    name: str
    operand: str
    order: tuple[int, ...]
    result: TensorSpec


@dataclass(frozen=True)
class StoreOp:
    dest: str
    indices: tuple[IntExpr, ...]
    tile: str


@dataclass(frozen=True)
class KernelArg:
    name: str
    memref: MemRefSpec


@dataclass(frozen=True)
class KernelGraph:
    name: str
    args: tuple[KernelArg, ...]
    ops: tuple[object, ...]
    uses_block_ids: bool


@dataclass(frozen=True)
class _MemRefValue:
    spec: MemRefSpec


@dataclass(frozen=True)
class _IntValue:
    expr: IntExpr


@dataclass(frozen=True)
class _FloatValue:
    value: float


@dataclass(frozen=True)
class _TileValue:
    spec: TensorSpec


_UNARY_OPS = {
    "cos",
    "exp",
    "reduce_max",
    "reduce_sum",
    "reciprocal",
    "rsqrt",
    "sigmoid",
    "silu",
    "sin",
    "tanh",
}

_UNARY_ALIASES = {
    "max": "reduce_max",
    "sum": "reduce_sum",
}


def parse_kernel(fn: object) -> KernelGraph:
    if not getattr(fn, "__xt_kernel__", False):
        raise XTConversionError("xt.convert(...) expects a function decorated with @xt.kernel")

    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except OSError as exc:
        raise XTConversionError(
            "xt.convert(...) requires a kernel defined in a real Python source file"
        ) from exc
    module = ast.parse(source)
    func_def = next(
        (node for node in module.body if isinstance(node, ast.FunctionDef)),
        None,
    )
    if func_def is None:
        raise XTConversionError("failed to locate function definition")

    signature = inspect.signature(fn)
    args: list[KernelArg] = []
    env: dict[str, object] = {}
    for name, param in signature.parameters.items():
        spec = MemRefSpec.parse(param.annotation)
        args.append(KernelArg(name=name, memref=spec))
        env[name] = _MemRefValue(spec)

    ops: list[object] = []
    uses_block_ids = False
    for stmt in func_def.body:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1:
                raise XTConversionError("only single-target assignments are supported")
            target = stmt.targets[0]
            value = stmt.value
            if isinstance(target, ast.Tuple):
                tuple_ops = _parse_tuple_assign(target, value, env)
                ops.extend(tuple_ops)
                continue
            if not isinstance(target, ast.Name):
                raise XTConversionError("only single-name assignments are supported")
            name = target.id
            if isinstance(value, ast.Constant) and isinstance(value.value, int):
                expr = IntExpr(kind="const", value=value.value, name=name)
                env[name] = _IntValue(expr)
                ops.append(ConstOp(name=name, value=value.value))
                continue
            if isinstance(value, ast.Constant) and isinstance(value.value, float):
                env[name] = _FloatValue(float(value.value))
                continue
            if _is_xt_call(value, "bid"):
                dim = _extract_int(value.args[0], env)
                env[name] = _IntValue(IntExpr(kind="bid", value=dim, name=name))
                uses_block_ids = True
                continue
            if _is_xt_call(value, "load"):
                load_op = _parse_load(name, value, env)
                env[name] = _TileValue(load_op.result)
                uses_block_ids |= any(expr.kind == "bid" for expr in load_op.indices)
                ops.append(load_op)
                continue
            if isinstance(value, ast.BinOp):
                binary_op = _parse_binop(name, value, env)
                if isinstance(binary_op, list):
                    final_op = binary_op[-1]
                    env[name] = _TileValue(final_op.result)
                    ops.extend(binary_op)
                else:
                    env[name] = _TileValue(binary_op.result)
                    ops.append(binary_op)
                continue
            if isinstance(value, ast.Call):
                op = _parse_call_assignment(name, value, env)
                env[name] = _TileValue(op.result)
                ops.append(op)
                continue
            raise XTConversionError(f"unsupported assignment for '{name}'")

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            if _is_xt_call(stmt.value, "store"):
                store_op = _parse_store(stmt.value, env)
                uses_block_ids |= any(expr.kind == "bid" for expr in store_op.indices)
                ops.append(store_op)
                continue
        raise XTConversionError(
            f"unsupported statement: {ast.dump(stmt, include_attributes=False)}"
        )

    return KernelGraph(
        name=func_def.name,
        args=tuple(args),
        ops=tuple(ops),
        uses_block_ids=uses_block_ids,
    )


def _parse_load(name: str, node: ast.Call, env: dict[str, object]) -> LoadOp:
    source = _expect_name(node.args[0], "xt.load source")
    source_value = _expect_memref(env, source)
    index_expr = _require_keyword(node, "index")
    shape_expr = _require_keyword(node, "shape")
    shared_node = _optional_keyword(node, "shared")
    indices = _extract_index_tuple(index_expr, env)
    shape = _extract_shape_tuple(shape_expr, env)
    shared = _extract_int(shared_node, env) if shared_node is not None else None
    return LoadOp(
        name=name,
        source=source,
        indices=indices,
        shape=shape,
        shared=shared,
        result=TensorSpec(shape=shape, element_type=source_value.spec.element_type),
    )


def _parse_tuple_assign(
    target: ast.Tuple, node: ast.AST, env: dict[str, object]
) -> list[ConstOp]:
    if not isinstance(node, ast.Tuple):
        raise XTConversionError("tuple assignment requires a tuple value")
    if len(target.elts) != len(node.elts):
        raise XTConversionError("tuple assignment target/value length mismatch")

    ops: list[ConstOp] = []
    for target_elt, value_elt in zip(target.elts, node.elts, strict=True):
        if not isinstance(target_elt, ast.Name):
            raise XTConversionError("tuple assignment targets must be variable names")
        if not isinstance(value_elt, ast.Constant) or not isinstance(value_elt.value, int):
            raise XTConversionError("tuple assignment only supports integer constants")
        expr = IntExpr(kind="const", value=value_elt.value, name=target_elt.id)
        env[target_elt.id] = _IntValue(expr)
        ops.append(ConstOp(name=target_elt.id, value=value_elt.value))
    return ops


def _parse_binop(name: str, node: ast.BinOp, env: dict[str, object]) -> object:
    lhs_name, lhs_value, lhs_ops = _materialize_binop_operand(
        name, "lhs", node.left, env
    )
    rhs_name, rhs_value, rhs_ops = _materialize_binop_operand(
        name, "rhs", node.right, env
    )
    prefix_ops = [*lhs_ops, *rhs_ops]

    lhs_name, lhs_value, rhs_name, rhs_value, promoted_ops = _promote_float_operands(
        name, lhs_name, lhs_value, rhs_name, rhs_value
    )

    lhs = _expect_tile(env_from_pair(lhs_name, lhs_value), lhs_name)
    rhs = _expect_tile(env_from_pair(rhs_name, rhs_value), rhs_name)

    op_map = {
        ast.Add: "add",
        ast.Sub: "sub",
        ast.Mult: "mul",
    }
    for ast_type, op_name in op_map.items():
        if isinstance(node.op, ast_type):
            result = lhs.spec.broadcast_with(rhs.spec)
            binary_op = BinaryOp(
                name=name,
                op_name=op_name,
                lhs=lhs_name,
                rhs=rhs_name,
                result=result,
            )
            if prefix_ops or promoted_ops:
                return [*prefix_ops, *promoted_ops, binary_op]
            return binary_op
    if isinstance(node.op, ast.MatMult):
        matmul_op = _parse_matmul(name, lhs_name, rhs_name, lhs.spec, rhs.spec)
        if prefix_ops or promoted_ops:
            return [*prefix_ops, *promoted_ops, matmul_op]
        return matmul_op
    raise XTConversionError(f"unsupported binary operator: {ast.dump(node.op)}")


def _materialize_binop_operand(
    parent_name: str, side: str, node: ast.AST, env: dict[str, object]
) -> tuple[str, object | None, list[object]]:
    if isinstance(node, ast.Name):
        return node.id, env.get(node.id), []
    if isinstance(node, ast.Constant) and isinstance(node.value, float):
        scalar_name = f"{parent_name}_{side}_scalar"
        scalar_value = _FloatValue(float(node.value))
        env[scalar_name] = scalar_value
        return scalar_name, scalar_value, []
    if isinstance(node, ast.BinOp):
        nested_name = f"{parent_name}_{side}"
        nested_op = _parse_binop(nested_name, node, env)
        nested_ops = nested_op if isinstance(nested_op, list) else [nested_op]
        final_op = nested_ops[-1]
        if not hasattr(final_op, "result"):
            raise XTConversionError("nested binary expression did not produce a tile")
        env[nested_name] = _TileValue(final_op.result)
        return nested_name, env[nested_name], nested_ops
    raise XTConversionError(
        f"unsupported binary operand: {ast.dump(node, include_attributes=False)}"
    )


def _promote_float_operands(
    name: str,
    lhs_name: str,
    lhs_value: object | None,
    rhs_name: str,
    rhs_value: object | None,
) -> tuple[str, object | None, str, object | None, list[FullOp]]:
    promoted: list[FullOp] = []

    if isinstance(lhs_value, _FloatValue) and isinstance(rhs_value, _TileValue):
        promoted_shape = (1,) * len(rhs_value.spec.shape)
        promoted_name = f"{name}_lhs_scalar"
        promoted_op = FullOp(
            name=promoted_name,
            shape=promoted_shape,
            value=lhs_value.value,
            result=TensorSpec(
                shape=promoted_shape,
                element_type=rhs_value.spec.element_type,
            ),
        )
        lhs_name = promoted_name
        lhs_value = _TileValue(promoted_op.result)
        promoted.append(promoted_op)

    if isinstance(rhs_value, _FloatValue) and isinstance(lhs_value, _TileValue):
        promoted_shape = (1,) * len(lhs_value.spec.shape)
        promoted_name = f"{name}_rhs_scalar"
        promoted_op = FullOp(
            name=promoted_name,
            shape=promoted_shape,
            value=rhs_value.value,
            result=TensorSpec(
                shape=promoted_shape,
                element_type=lhs_value.spec.element_type,
            ),
        )
        rhs_name = promoted_name
        rhs_value = _TileValue(promoted_op.result)
        promoted.append(promoted_op)

    return lhs_name, lhs_value, rhs_name, rhs_value, promoted


def env_from_pair(name: str, value: object | None) -> dict[str, object]:
    return {} if value is None else {name: value}


def _parse_matmul(
    name: str,
    lhs_name: str,
    rhs_name: str,
    lhs: TensorSpec,
    rhs: TensorSpec,
) -> MatmulOp:
    if len(lhs.shape) != 2 or len(rhs.shape) != 2:
        raise XTConversionError("matmul requires rank-2 tensor operands")
    if lhs.element_type != rhs.element_type:
        raise XTConversionError(
            f"matmul element type mismatch: {lhs.element_type} vs {rhs.element_type}"
        )
    if lhs.shape[1] != rhs.shape[0]:
        raise XTConversionError(
            "matmul requires lhs inner dimension to match rhs outer dimension"
        )
    return MatmulOp(
        name=name,
        lhs=lhs_name,
        rhs=rhs_name,
        result=TensorSpec(
            shape=(lhs.shape[0], rhs.shape[1]),
            element_type=lhs.element_type,
        ),
    )


def _parse_call_assignment(name: str, node: ast.Call, env: dict[str, object]) -> object:
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        if node.func.value.id != "xt":
            raise XTConversionError("only xt.* calls are supported")
        op_name = node.func.attr
        op_name = _UNARY_ALIASES.get(op_name, op_name)
        if op_name in _UNARY_OPS:
            operand_name = _expect_name(node.args[0], f"xt.{op_name} operand")
            operand = _expect_tile(env, operand_name)
            result = (
                operand.spec.reduce_last_dim()
                if op_name in {"reduce_sum", "reduce_max"}
                else operand.spec
            )
            return UnaryOp(
                name=name,
                op_name=op_name,
                operand=operand_name,
                result=result,
            )
        if op_name == "reshape":
            operand_name = _expect_name(node.args[0], "xt.reshape operand")
            operand = _expect_tile(env, operand_name)
            shape = _extract_shape_tuple(_require_keyword(node, "shape"), env)
            return ReshapeOp(
                name=name,
                operand=operand_name,
                shape=shape,
                result=operand.spec.reshape(shape),
            )
        if op_name == "transpose":
            operand_name = _expect_name(node.args[0], "xt.transpose operand")
            operand = _expect_tile(env, operand_name)
            order_node = _optional_keyword(node, "order")
            if order_node is None:
                rank = len(operand.spec.shape)
                if rank < 2:
                    raise XTConversionError("transpose requires rank >= 2")
                order = tuple(range(rank - 2)) + (rank - 1, rank - 2)
            else:
                order = _extract_shape_tuple(order_node, env)
            return TransposeOp(
                name=name,
                operand=operand_name,
                order=order,
                result=operand.spec.transpose(order),
            )
        if op_name == "full":
            shape = _extract_shape_tuple(_require_keyword(node, "shape"), env)
            value = _extract_float(_require_keyword(node, "value"))
            return FullOp(
                name=name,
                shape=shape,
                value=value,
                result=TensorSpec(shape=shape, element_type="f32"),
            )
    raise XTConversionError(
        f"unsupported xt call: {ast.dump(node, include_attributes=False)}"
    )


def _parse_store(node: ast.Call, env: dict[str, object]) -> StoreOp:
    dest = _expect_name(node.args[0], "xt.store destination")
    _expect_memref(env, dest)
    tile_name = _expect_name(_require_keyword(node, "tile"), "xt.store tile")
    _expect_tile(env, tile_name)
    indices = _extract_index_tuple(_require_keyword(node, "index"), env)
    return StoreOp(dest=dest, indices=indices, tile=tile_name)


def _extract_index_tuple(node: ast.AST, env: dict[str, object]) -> tuple[IntExpr, ...]:
    if not isinstance(node, ast.Tuple):
        raise XTConversionError("index= must be a tuple")
    return tuple(_extract_int_expr(elt, env) for elt in node.elts)


def _extract_shape_tuple(node: ast.AST, env: dict[str, object]) -> tuple[int, ...]:
    if not isinstance(node, ast.Tuple):
        raise XTConversionError("shape/order= must be a tuple")
    return tuple(_extract_int(elt, env) for elt in node.elts)


def _extract_int_expr(node: ast.AST, env: dict[str, object]) -> IntExpr:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return IntExpr(kind="const", value=node.value)
    if isinstance(node, ast.Name):
        value = env.get(node.id)
        if isinstance(value, _IntValue):
            return value.expr
    if _is_xt_call(node, "bid"):
        return IntExpr(kind="bid", value=_extract_int(node.args[0], env))
    raise XTConversionError(f"expected integer or xt.bid(...), got {ast.dump(node)}")


def _extract_int(node: ast.AST | None, env: dict[str, object]) -> int:
    if node is None:
        raise XTConversionError("expected integer value")
    expr = _extract_int_expr(node, env)
    return expr.value


def _extract_float(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    raise XTConversionError(f"expected float constant, got {ast.dump(node)}")


def _expect_name(node: ast.AST, label: str) -> str:
    if not isinstance(node, ast.Name):
        raise XTConversionError(f"{label} must be a variable name")
    return node.id


def _expect_memref(env: dict[str, object], name: str) -> _MemRefValue:
    value = env.get(name)
    if not isinstance(value, _MemRefValue):
        raise XTConversionError(f"'{name}' is not a memref value")
    return value


def _expect_tile(env: dict[str, object], name: str) -> _TileValue:
    value = env.get(name)
    if not isinstance(value, _TileValue):
        raise XTConversionError(f"'{name}' is not a tile value")
    return value


def _is_xt_call(node: ast.AST, attr: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "xt"
        and node.func.attr == attr
    )


def _require_keyword(node: ast.Call, name: str) -> ast.AST:
    value = _optional_keyword(node, name)
    if value is None:
        raise XTConversionError(f"missing required keyword argument: {name}")
    return value


def _optional_keyword(node: ast.Call, name: str) -> ast.AST | None:
    for keyword in node.keywords:
        if keyword.arg == name:
            return keyword.value
    return None
