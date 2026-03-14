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
_MATMUL_OP_RE = re.compile(
    r'^(?P<indent>\s*)(?P<result>%[\w\d._]+) = "xt\.matmul"\((?P<lhs>%[^,]+), (?P<rhs>%[^)]+)\) : '
    r'\((?P<lhs_type>tensor<[^>]+>), (?P<rhs_type>tensor<[^>]+>)\) -> '
    r'(?P<result_type>tensor<[^>]+>)$',
    re.MULTILINE,
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
_NOVA_MATMUL_LINE_RE = re.compile(
    r'^(?P<indent>\s*)(?P<result>%[\w\d._]+) = "nova\.matmul"'
    r'\((?P<lhs>%[^,]+), (?P<rhs>%[^,]+), (?P<scale>%[^,]+), (?P<bias>%[^)]+)\) : '
    r'\((?P<lhs_type>tensor<[^>]+>), (?P<rhs_type>tensor<[^>]+>), '
    r'(?P<scale_type>tensor<[^>]+>), (?P<bias_type>tensor<[^>]+>)\) -> '
    r'(?P<result_type>tensor<[^>]+>)$'
)
_PURE_RESULT_LINE_RE = re.compile(
    r'^\s*(?P<result>%[\w\d._]+)\s*=\s*(?:"nova\.[^"]+"|arith\.constant(?:\s|$))'
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
    rewritten = _MATMUL_OP_RE.sub(_rewrite_matmul_op, rewritten)
    if rewritten == text:
        return module

    ctx = module.context
    ctx.allow_unregistered_dialects = True
    with ctx:
        return ir.Module.parse(rewritten)


def optimize_nova(module: ir.Module) -> ir.Module:
    text = str(module)
    rewritten = text
    while True:
        next_rewritten = _rewrite_nova_scalar_folds(rewritten)
        if next_rewritten == rewritten:
            break
        rewritten = next_rewritten
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


def _rewrite_matmul_op(match: re.Match[str]) -> str:
    result_suffix = match.group("result").lstrip("%").replace(".", "_")
    scale_name = f"%matmul_scale_{result_suffix}"
    bias_name = f"%matmul_bias_{result_suffix}"
    indent = match.group("indent")
    return (
        f"{indent}{scale_name} = arith.constant dense<1.000000e+00> : tensor<1x1xf32>\n"
        f"{indent}{bias_name} = arith.constant dense<0.000000e+00> : tensor<1x1xf32>\n"
        f'{indent}{match.group("result")} = "nova.matmul"({match.group("lhs")}, {match.group("rhs")}, '
        f"{scale_name}, {bias_name}) : "
        f'({match.group("lhs_type")}, {match.group("rhs_type")}, tensor<1x1xf32>, tensor<1x1xf32>) -> '
        f'{match.group("result_type")}'
    )


def _rewrite_nova_scalar_folds(text: str) -> str:
    lines = text.splitlines()
    scalar_defs: dict[str, dict[str, object]] = {}
    constant_defs: dict[str, float] = {}
    matmul_defs: dict[str, dict[str, str]] = {}
    for index, line in enumerate(lines):
        match = _NOVA_SCALAR_LINE_RE.match(line)
        if match is not None:
            scalar_defs[match.group("result")] = {
                "result": match.group("result"),
                "index": index,
                "input": match.group("input"),
                "mode": int(match.group("mode")),
                "rhs": float(match.group("rhs")),
            }
            continue

        matmul_match = _NOVA_MATMUL_LINE_RE.match(line)
        if matmul_match is not None:
            matmul_defs[matmul_match.group("result")] = {
                "lhs": matmul_match.group("lhs"),
                "rhs": matmul_match.group("rhs"),
                "scale": matmul_match.group("scale"),
                "bias": matmul_match.group("bias"),
                "lhs_type": matmul_match.group("lhs_type"),
                "rhs_type": matmul_match.group("rhs_type"),
                "scale_type": matmul_match.group("scale_type"),
                "bias_type": matmul_match.group("bias_type"),
                "result_type": matmul_match.group("result_type"),
            }
            continue

        constant_match = _CONSTANT_OP_RE.match(line.strip())
        if constant_match is not None and _is_scalar_tensor_type(
            constant_match.group("type")
        ):
            constant_defs[constant_match.group("name")] = float(
                constant_match.group("value")
            )

    changed = False
    for index, line in enumerate(lines):
        scalar_match = _NOVA_SCALAR_LINE_RE.match(line)
        if scalar_match is not None:
            rewritten = _rewrite_nova_scalar_over_matmul_line(
                scalar_match, matmul_defs, constant_defs
            )
            if rewritten != line:
                lines[index] = rewritten
                changed = True
            continue

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
        pure_match = _PURE_RESULT_LINE_RE.match(line)
        if pure_match is not None and pure_match.group("result") not in used_values:
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


def _rewrite_nova_scalar_over_matmul_line(
    match: re.Match[str],
    matmul_defs: dict[str, dict[str, str]],
    constant_defs: dict[str, float],
) -> str:
    matmul = matmul_defs.get(match.group("input"))
    if matmul is None:
        return match.group(0)
    scale_name = matmul["scale"]
    bias_name = matmul["bias"]
    scale_value = constant_defs.get(scale_name)
    bias_value = constant_defs.get(bias_name)
    if scale_value is None or bias_value is None:
        return match.group(0)

    mode = int(match.group("mode"))
    rhs = float(match.group("rhs"))
    if mode == 1:
        new_scale = scale_value
        new_bias = bias_value + rhs
    elif mode == 2:
        new_scale = scale_value * rhs
        new_bias = bias_value * rhs
    else:
        return match.group(0)

    indent = match.group("indent")
    result_suffix = match.group("result").lstrip("%").replace(".", "_")
    new_scale_name = f"%matmul_scale_folded_{result_suffix}"
    new_bias_name = f"%matmul_bias_folded_{result_suffix}"
    return (
        f"{indent}{new_scale_name} = arith.constant dense<{new_scale:.6e}> : {matmul['scale_type']}\n"
        f"{indent}{new_bias_name} = arith.constant dense<{new_bias:.6e}> : {matmul['bias_type']}\n"
        f'{indent}{match.group("result")} = "nova.matmul"({matmul["lhs"]}, {matmul["rhs"]}, {new_scale_name}, {new_bias_name}) : '
        f'({matmul["lhs_type"]}, {matmul["rhs_type"]}, {matmul["scale_type"]}, {matmul["bias_type"]}) -> '
        f'{matmul["result_type"]}'
    )


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
