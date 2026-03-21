#!/usr/bin/env python3

import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "kernels" / "matmul.py"


def load_module():
    fake_xt = types.ModuleType("xtile")
    fake_xt.int8 = "int8"
    fake_xt.float32 = "float32"
    fake_xt.kernel = lambda fn: fn
    fake_xt.bid = lambda dim: 0
    fake_xt.load = lambda *args, **kwargs: None
    fake_xt.matmul = lambda lhs, rhs: None
    fake_xt.astype = lambda value, dtype: value
    fake_xt.store = lambda *args, **kwargs: None
    fake_xt.compile = lambda ir: ir

    class Array:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype
            self.ndim = len(shape)

    fake_xt.Array = Array
    fake_xt.convert = lambda *args, **kwargs: "ir"
    fake_xt.saved_path = None

    def save_compile_results(ir, path):
        fake_xt.saved_path = path

    fake_xt.save_compile_results = save_compile_results

    sys.modules["xtile"] = fake_xt

    spec = importlib.util.spec_from_file_location("matmul_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, fake_xt


def main():
    module, fake_xt = load_module()

    args = module.parse_args([])
    if (args.m, args.n, args.k) != (128, 128, 256):
        raise AssertionError(f"unexpected defaults: {(args.m, args.n, args.k)}")

    module.main(["--m", "64", "--n", "32", "--k", "16"])
    if fake_xt.saved_path != "compiled/matmul/matmul_64x32x16":
        raise AssertionError(f"unexpected output path: {fake_xt.saved_path}")

    print("matmul CLI path test passed")


if __name__ == "__main__":
    main()
