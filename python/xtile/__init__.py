from __future__ import annotations

from mlir import ir

from .ast_parser import parse_kernel
from .dsl import (
    bid,
    cos,
    exp,
    kernel,
    load,
    max,
    reciprocal,
    reshape,
    rsqrt,
    sigmoid,
    silu,
    sin,
    store,
    sum,
    tanh,
    transpose,
)
from .ir_builder import build_module
from .types import memref


def convert(fn: object) -> ir.Module:
    return build_module(parse_kernel(fn))


def dump(module: ir.Module) -> str:
    return str(module)


__all__ = [
    "bid",
    "cos",
    "convert",
    "dump",
    "exp",
    "kernel",
    "load",
    "max",
    "memref",
    "reciprocal",
    "reshape",
    "rsqrt",
    "sigmoid",
    "silu",
    "sin",
    "store",
    "sum",
    "tanh",
    "transpose",
]
