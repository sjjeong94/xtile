from __future__ import annotations

import re

from mlir import ir

_BINARY_OP_RE = re.compile(
    r'"xt\.(?P<op>add|mul|sub)"\((?P<lhs>%[^,]+), (?P<rhs>%[^)]+)\) : '
    r'\((?P<lhs_type>tensor<[^>]+>), (?P<rhs_type>tensor<[^>]+>)\) -> '
    r'(?P<result_type>tensor<[^>]+>)'
)
_CONSTANT_OP_RE = re.compile(
    r'(?P<name>%[\w\d._]+) = arith.constant dense<(?P<value>[-+0-9.eE]+)> : '
    r'(?P<type>tensor<[^>]+>)'
)
_REDUCE_OP_RE = re.compile(
    r'"xt\.(?P<op>reduce_sum|reduce_max)"\((?P<input>%[^)]+)\) : '
    r'\((?P<input_type>tensor<[^>]+>)\) -> (?P<result_type>tensor<[^>]+>)'
)
_MODE_BY_OP = {"add": 1, "mul": 2, "sub": 3}
_REDUCE_MODE_BY_OP = {"reduce_sum": 0, "reduce_max": 1}
_DEFAULT_ATTRS = (
    "lhs_b = 0.000000e+00 : f32, "
    "lhs_s = 1.000000e+00 : f32, "
    "mode = {mode} : i32, "
    "rhs_b = 0.000000e+00 : f32, "
    "rhs_s = 1.000000e+00 : f32"
)


def to_nova(module: ir.Module) -> ir.Module:
    text = str(module)
    rewritten = _REDUCE_OP_RE.sub(_rewrite_reduce_op, text)
    constant_map = {
        match.group("name"): float(match.group("value"))
        for match in _CONSTANT_OP_RE.finditer(rewritten)
        if _is_scalar_tensor_type(match.group("type"))
    }
    rewritten = _BINARY_OP_RE.sub(
        lambda match: _rewrite_binary_op(match, constant_map), rewritten
    )
    if rewritten == text:
        return module

    ctx = module.context
    ctx.allow_unregistered_dialects = True
    with ctx:
        return ir.Module.parse(rewritten)


def _rewrite_binary_op(
    match: re.Match[str], constant_map: dict[str, float]
) -> str:
    lhs_type = match.group("lhs_type")
    rhs_type = match.group("rhs_type")
    result_type = match.group("result_type")
    rhs_name = match.group("rhs")
    rhs_constant = constant_map.get(rhs_name)
    if rhs_constant is not None:
        mode = _MODE_BY_OP[match.group("op")]
        return (
            f'"nova.scalar"({match.group("lhs")}) '
            f'<{{mode = {mode} : i32, rhs = {rhs_constant:.6e} : f32}}> : '
            f'({lhs_type}) -> {result_type}'
        )
    if _is_scalar_tensor_type(lhs_type) or _is_scalar_tensor_type(rhs_type):
        return match.group(0)
    nova_op = (
        "nova.elementwise"
        if lhs_type == result_type and rhs_type == result_type
        else "nova.broadcast"
    )
    mode = _MODE_BY_OP[match.group("op")]
    attrs = _DEFAULT_ATTRS.format(mode=mode)
    return (
        f'"{nova_op}"({match.group("lhs")}, {match.group("rhs")}) '
        f'<{{{attrs}}}> : ({lhs_type}, {rhs_type}) -> {result_type}'
    )


def _rewrite_reduce_op(match: re.Match[str]) -> str:
    mode = _REDUCE_MODE_BY_OP[match.group("op")]
    return (
        f'"nova.reduce"({match.group("input")}) '
        f'<{{mode = {mode} : i32}}> : '
        f'({match.group("input_type")}) -> {match.group("result_type")}'
    )


def _is_scalar_tensor_type(type_text: str) -> bool:
    body = type_text[len("tensor<") : -1]
    dims_part, _sep, _elem = body.rpartition("x")
    dims = [int(token) for token in dims_part.split("x") if token]
    num_elements = 1
    for dim in dims:
        num_elements *= dim
    return num_elements == 1
