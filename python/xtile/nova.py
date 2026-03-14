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
_NOVA_SCALAR_LINE_RE = re.compile(
    r'^(?P<indent>\s*)(?P<result>%[\w\d._]+) = "nova\.scalar"\((?P<input>%[^)]+)\) '
    r'<\{mode = (?P<mode>\d+) : i32, rhs = (?P<rhs>[-+0-9.eE]+) : f32\}> : '
    r'\((?P<input_type>tensor<[^>]+>)\) -> (?P<result_type>tensor<[^>]+>)$'
)
_NOVA_BINARY_LINE_RE = re.compile(
    r'^(?P<indent>\s*)(?P<result>%[\w\d._]+) = "(?P<op>nova\.(?:broadcast|elementwise))"'
    r'\((?P<lhs>%[^,]+), (?P<rhs>%[^)]+)\) '
    r'<\{lhs_b = (?P<lhs_b>[-+0-9.eE]+) : f32, lhs_s = (?P<lhs_s>[-+0-9.eE]+) : f32, '
    r'mode = (?P<mode>\d+) : i32, rhs_b = (?P<rhs_b>[-+0-9.eE]+) : f32, '
    r'rhs_s = (?P<rhs_s>[-+0-9.eE]+) : f32\}> : '
    r'\((?P<lhs_type>tensor<[^>]+>), (?P<rhs_type>tensor<[^>]+>)\) -> '
    r'(?P<result_type>tensor<[^>]+>)$'
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


def optimize_nova(module: ir.Module) -> ir.Module:
    text = str(module)
    rewritten = _rewrite_nova_scalar_folds(text)
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
        if mode == 3:
            mode = 1
            rhs_constant = -rhs_constant
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


def _rewrite_nova_scalar_folds(text: str) -> str:
    lines = text.splitlines()
    scalar_defs: dict[str, dict[str, object]] = {}
    for index, line in enumerate(lines):
        match = _NOVA_SCALAR_LINE_RE.match(line)
        if match is None:
            continue
        scalar_defs[match.group("result")] = {
            "index": index,
            "input": match.group("input"),
            "mode": int(match.group("mode")),
            "rhs": float(match.group("rhs")),
        }

    changed = False
    for index, line in enumerate(lines):
        match = _NOVA_BINARY_LINE_RE.match(line)
        if match is None:
            continue
        rewritten = _rewrite_nova_binary_line(match, scalar_defs)
        if rewritten != line:
            lines[index] = rewritten
            changed = True

    if not changed:
        return text

    used_values = _collect_ssa_uses(lines)
    filtered_lines = []
    for line in lines:
        match = _NOVA_SCALAR_LINE_RE.match(line)
        if match is not None and match.group("result") not in used_values:
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines) + ("\n" if text.endswith("\n") else "")


def _rewrite_nova_binary_line(
    match: re.Match[str], scalar_defs: dict[str, dict[str, object]]
) -> str:
    lhs = match.group("lhs")
    rhs = match.group("rhs")
    lhs_bias = float(match.group("lhs_b"))
    lhs_scale = float(match.group("lhs_s"))
    rhs_bias = float(match.group("rhs_b"))
    rhs_scale = float(match.group("rhs_s"))
    changed = False

    lhs_scalar = scalar_defs.get(lhs)
    if lhs_scalar is not None:
        folded = _fold_scalar_attrs(
            lhs_scale, lhs_bias, int(lhs_scalar["mode"]), float(lhs_scalar["rhs"])
        )
        if folded is not None:
            lhs_scale, lhs_bias = folded
            lhs = str(lhs_scalar["input"])
            changed = True

    rhs_scalar = scalar_defs.get(rhs)
    if rhs_scalar is not None:
        folded = _fold_scalar_attrs(
            rhs_scale, rhs_bias, int(rhs_scalar["mode"]), float(rhs_scalar["rhs"])
        )
        if folded is not None:
            rhs_scale, rhs_bias = folded
            rhs = str(rhs_scalar["input"])
            changed = True

    if not changed:
        return match.group(0)

    return (
        f'{match.group("indent")}{match.group("result")} = "{match.group("op")}"'
        f"({lhs}, {rhs}) "
        f'<{{lhs_b = {lhs_bias:.6e} : f32, lhs_s = {lhs_scale:.6e} : f32, '
        f'mode = {match.group("mode")} : i32, rhs_b = {rhs_bias:.6e} : f32, '
        f'rhs_s = {rhs_scale:.6e} : f32}}> : '
        f'({match.group("lhs_type")}, {match.group("rhs_type")}) -> '
        f'{match.group("result_type")}'
    )


def _fold_scalar_attrs(
    scale: float, bias: float, mode: int, rhs: float
) -> tuple[float, float] | None:
    if mode == 1:
        return scale, bias + rhs * scale
    if mode == 2:
        return scale * rhs, bias
    return None


def _collect_ssa_uses(lines: list[str]) -> set[str]:
    used_values: set[str] = set()
    for line in lines:
        if "=" in line:
            _def, rest = line.split("=", 1)
        else:
            rest = line
        used_values.update(re.findall(r"%[\w\d._]+(?:#\d+)?", rest))
    return used_values


def _is_scalar_tensor_type(type_text: str) -> bool:
    body = type_text[len("tensor<") : -1]
    dims_part, _sep, _elem = body.rpartition("x")
    dims = [int(token) for token in dims_part.split("x") if token]
    num_elements = 1
    for dim in dims:
        num_elements *= dim
    return num_elements == 1
