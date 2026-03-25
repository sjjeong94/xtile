from __future__ import annotations

from importlib import import_module
import importlib.util
from pathlib import Path
import os
import sys
from typing import Any

from .dsl import (
    Array,
    add,
    astype,
    bid,
    convert as _convert_impl,
    cos,
    exp,
    float32,
    int8,
    kernel,
    load,
    load_conv2d,
    matmul,
    max,
    mma,
    mul,
    reciprocal,
    reshape,
    rsqrt,
    sigmoid,
    silu,
    sin,
    store,
    sub,
    sum,
    tanh,
    transpose,
)


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _load_native_module():
    try:
        return import_module("xtile._xtile")
    except ModuleNotFoundError:
        package_dir = Path(__file__).resolve().parent
        candidate_dirs = [
            package_dir,
            package_dir.parent / "build" / "python" / "xtile",
        ]
        for candidate_dir in candidate_dirs:
            for pattern in ("_xtile*.so", "_xtile*.pyd"):
                matches = sorted(candidate_dir.glob(pattern))
                if not matches:
                    continue
                spec = importlib.util.spec_from_file_location(
                    "xtile._xtile", matches[0]
                )
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules["xtile._xtile"] = module
                spec.loader.exec_module(module)
                return module
        raise


_native = _load_native_module()


def _wrap_module_asm(asm: str) -> tuple[str, str | None]:
    stripped = asm.strip()
    if stripped.startswith("module"):
        return stripped, None

    indented = "\n".join(
        f"  {line}" if line else line for line in stripped.splitlines()
    )
    wrapped = f"module {{\n{indented}\n}}"
    return wrapped, stripped


class Module:
    def __init__(self, raw_module: Any, display_asm: str | None = None) -> None:
        self._raw_module = raw_module
        self._display_asm = display_asm

    @property
    def _CAPIPtr(self):
        return self._raw_module

    def __str__(self) -> str:
        if self._display_asm is not None:
            return self._display_asm
        return _native._module_asm(self)

    def _clear_display_override(self) -> None:
        self._display_asm = None


def _normalize_module_object(module: Any) -> Any:
    return module


def _clear_display_if_wrapped(module: Any) -> None:
    if isinstance(module, Module):
        module._clear_display_override()


def parse(asm: str) -> Module:
    wrapped_asm, display_asm = _wrap_module_asm(asm)
    return Module(_native.parse(wrapped_asm), display_asm=display_asm)


def xt_to_nova(module: Any):
    result = _native.xt_to_nova(_normalize_module_object(module))
    _clear_display_if_wrapped(module)
    return module if isinstance(module, Module) else result


def xt_serialize(module: Any):
    result = _native.xt_serialize(_normalize_module_object(module))
    _clear_display_if_wrapped(module)
    return module if isinstance(module, Module) else result


def nova_optimize(module: Any):
    result = _native.nova_optimize(_normalize_module_object(module))
    _clear_display_if_wrapped(module)
    return module if isinstance(module, Module) else result


def nova_allocate(module: Any):
    result = _native.nova_allocate(_normalize_module_object(module))
    _clear_display_if_wrapped(module)
    return module if isinstance(module, Module) else result


def nova_threading(module: Any):
    result = _native.nova_threading(_normalize_module_object(module))
    _clear_display_if_wrapped(module)
    return module if isinstance(module, Module) else result


def nova_barrier(module: Any):
    result = _native.nova_barrier(_normalize_module_object(module))
    _clear_display_if_wrapped(module)
    return module if isinstance(module, Module) else result


def nova_to_x1(module: Any):
    result = _native.nova_to_x1(_normalize_module_object(module))
    _clear_display_if_wrapped(module)
    return module if isinstance(module, Module) else result


def compile(module: Any):
    module = xt_serialize(module)
    module = xt_to_nova(module)
    module = nova_optimize(module)
    module = nova_threading(module)
    module = nova_barrier(module)
    module = nova_allocate(module)
    module = nova_to_x1(module)
    return module


def convert(kernel_fn, *, args, grid, double_buffering=False):
    return _convert_impl(
        kernel_fn,
        args=args,
        grid=grid,
        double_buffering=double_buffering,
        parse_module=parse,
    )


def save_ir(ir: object, save_path: str) -> None:
    with open(save_path, "wt") as f:
        f.write(str(ir))


def save_compile_results(ir: object, save_dir: str) -> None:

    os.makedirs(save_dir, exist_ok=True)

    save_ir(ir, os.path.join(save_dir, "0_xt.mlir"))
    ir = xt_serialize(ir)
    save_ir(ir, os.path.join(save_dir, "1_xt_serialize.mlir"))
    ir = xt_to_nova(ir)
    save_ir(ir, os.path.join(save_dir, "2_xt_to_nova.mlir"))
    ir = nova_optimize(ir)
    save_ir(ir, os.path.join(save_dir, "3_nova_optimize.mlir"))
    ir = nova_threading(ir)
    save_ir(ir, os.path.join(save_dir, "4_nova_threading.mlir"))
    ir = nova_barrier(ir)
    save_ir(ir, os.path.join(save_dir, "5_nova_barrier.mlir"))
    ir = nova_allocate(ir)
    save_ir(ir, os.path.join(save_dir, "6_nova_allocate.mlir"))
    ir = nova_to_x1(ir)
    save_ir(ir, os.path.join(save_dir, "7_nova_to_x1.mlir"))
    print("compiled kernel ->", save_dir)


def _module_asm(module: Any) -> str:
    return _native._module_asm(_normalize_module_object(module))


def _parse_module(asm: str) -> Module:
    return parse(asm)


__all__ = [
    "Array",
    "Module",
    "add",
    "astype",
    "bid",
    "compile",
    "convert",
    "cos",
    "float32",
    "int8",
    "kernel",
    "load",
    "load_conv2d",
    "matmul",
    "max",
    "mma",
    "mul",
    "nova_allocate",
    "nova_barrier",
    "nova_optimize",
    "nova_to_x1",
    "nova_threading",
    "parse",
    "reciprocal",
    "reshape",
    "rsqrt",
    "xt_serialize",
    "sigmoid",
    "silu",
    "sin",
    "store",
    "sub",
    "sum",
    "tanh",
    "xt_to_nova",
    "transpose",
    "exp",
]
