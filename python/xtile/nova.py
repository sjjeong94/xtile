from __future__ import annotations

import re

from mlir import ir

_BINARY_OP_RE = re.compile(
    r'"xt\.(?P<op>add|mul|sub)"\((?P<lhs>%[^,]+), (?P<rhs>%[^)]+)\) : '
    r'\((?P<lhs_type>tensor<[^>]+>), (?P<rhs_type>tensor<[^>]+>)\) -> '
    r'(?P<result_type>tensor<[^>]+>)'
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
    rewritten = _BINARY_OP_RE.sub(_rewrite_binary_op, rewritten)
    if rewritten == text:
        return module

    ctx = module.context
    ctx.allow_unregistered_dialects = True
    with ctx:
        return ir.Module.parse(rewritten)


def _rewrite_binary_op(match: re.Match[str]) -> str:
    lhs_type = match.group("lhs_type")
    rhs_type = match.group("rhs_type")
    result_type = match.group("result_type")
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
