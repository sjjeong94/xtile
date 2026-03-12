from __future__ import annotations

from mlir import ir

from .ast_parser import parse_kernel
from .dsl import bid, exp, kernel, load, reshape, store, transpose
from .ir_builder import build_module
from .types import memref


def convert(fn: object) -> ir.Module:
    return build_module(parse_kernel(fn))


def dump(module: ir.Module) -> str:
    return str(module)


__all__ = [
    "bid",
    "convert",
    "dump",
    "exp",
    "kernel",
    "load",
    "memref",
    "reshape",
    "store",
    "transpose",
]
