#!/usr/bin/env python3

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) in sys.path:
    sys.path.remove(str(THIS_DIR))

import importlib.util
import types


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "kernels" / "conv2d.py"


def load_module():
    fake_xt = types.ModuleType("xtile")
    fake_xt.int8 = "int8"
    fake_xt.float32 = "float32"
    fake_xt.kernel = lambda fn: fn
    fake_xt.bid = lambda dim: 0
    fake_xt.cdiv = lambda a, b: (a + b - 1) // b
    fake_xt.load = lambda *args, **kwargs: None
    fake_xt.load_conv2d = lambda *args, **kwargs: None
    fake_xt.store = lambda *args, **kwargs: None

    class Array:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype
            self.ndim = len(shape)

    fake_xt.Array = Array
    fake_xt.convert = lambda *args, **kwargs: "ir"
    fake_xt.saved = []

    def save_ir(ir, path):
        fake_xt.saved.append(path)

    fake_xt.save_ir = save_ir
    fake_xt.xt_serialize = lambda ir: ir
    fake_xt.xt_to_nova = lambda ir: ir
    fake_xt.nova_optimize = lambda ir: ir
    fake_xt.nova_threading = lambda ir: ir
    fake_xt.nova_barrier = lambda ir: ir
    fake_xt.nova_allocate = lambda ir: ir
    fake_xt.nova_to_x1 = lambda ir: ir

    sys.modules["xtile"] = fake_xt

    spec = importlib.util.spec_from_file_location("conv2d_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, fake_xt


def main():
    module, fake_xt = load_module()

    args = module.parse_args([])
    if (args.n, args.h, args.w, args.cin, args.cout, args.kh, args.kw) != (
        1,
        32,
        64,
        128,
        64,
        3,
        3,
    ):
        raise AssertionError(
            f"unexpected defaults: {(args.n, args.h, args.w, args.cin, args.cout, args.kh, args.kw)}"
        )

    module.main(["--n", "1", "--h", "32", "--w", "64", "--cin", "128", "--cout", "64", "--kh", "3", "--kw", "3"])
    expected_dir = "compiled/conv2d/conv2d_1x32x64x128x64x3x3"
    expected_files = [
        f"{expected_dir}/0_xt.mlir",
        f"{expected_dir}/1_xt_serialize.mlir",
        f"{expected_dir}/2_xt_to_nova.mlir",
        f"{expected_dir}/3_nova_optimize.mlir",
        f"{expected_dir}/4_nova_threading.mlir",
        f"{expected_dir}/5_nova_barrier.mlir",
        f"{expected_dir}/6_nova_allocate.mlir",
        f"{expected_dir}/7_nova_to_x1.mlir",
    ]
    if fake_xt.saved != expected_files:
        raise AssertionError(f"unexpected saved outputs: {fake_xt.saved}")

    print("conv2d CLI path test passed")


if __name__ == "__main__":
    main()
