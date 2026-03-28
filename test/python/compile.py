#!/usr/bin/env python3

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) in sys.path:
    sys.path.remove(str(THIS_DIR))

import xtile as xt


def main():
    module = xt.parse(
        """
        module {
          func.func @compile_pipeline(%src: memref<128x16xf32>, %dst: memref<128x16xf32>) attributes {xt.grid = array<i32: 2, 1, 1>} {
            %bid:3 = xt.get_tile_block_id : i32, i32, i32
            %0 = xt.load(%src, %bid#0, %bid#1) : (memref<128x16xf32>, i32, i32) -> tensor<16x16xf32>
            %1 = xt.mul(%0, %0) : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>
            xt.store(%1, %dst, %bid#0, %bid#1) : (tensor<16x16xf32>, memref<128x16xf32>, i32, i32) -> ()
            func.return
          }
        }
        """
    )

    compiled = xt.compile(module)
    if compiled is not module:
        raise AssertionError("xt.compile should return the original module object")

    module_asm = xt._module_asm(module)
    expected_snippets = (
        "x1.load ",
        "x1.square ",
        "x1.barrier 0",
        "x1.barrier 1",
        "x1.store ",
        "space 3",
        "[8, 16]",
    )
    for snippet in expected_snippets:
        if snippet not in module_asm:
            raise AssertionError(f"expected {snippet!r} in compiled IR:\n{module_asm}")

    print(module_asm)

    conv2d_module = xt.parse(
        """
        module {
          func.func @compile_conv2d(%src: memref<1x32x64x128xi8>, %filter: memref<3x3x128x64xi8>, %dst: memref<1x32x64x64xf32>) attributes {xt.grid = array<i32: 1, 1, 1>} {
            %c0 = arith.constant 0 : i32
            %0 = xt.load(%filter, %c0, %c0, %c0, %c0) {shared = 2 : i64} : (memref<3x3x128x64xi8>, i32, i32, i32, i32) -> tensor<3x3x128x64xi8>
            %1 = xt.load_conv2d(%src, %0, %c0, %c0, %c0, %c0) {dilation = array<i64: 1, 1>, group = 1 : i64, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (memref<1x32x64x128xi8>, tensor<3x3x128x64xi8>, i32, i32, i32, i32) -> tensor<1x32x64x64xf32>
            xt.store(%1, %dst, %c0, %c0, %c0, %c0) : (tensor<1x32x64x64xf32>, memref<1x32x64x64xf32>, i32, i32, i32, i32) -> ()
            func.return
          }
        }
        """
    )

    compiled = xt.compile(conv2d_module)
    if compiled is not conv2d_module:
        raise AssertionError("xt.compile should return the original module object")

    conv2d_asm = xt._module_asm(conv2d_module)
    conv2d_expected_snippets = (
        "x1.load ",
        "x1.conv2d inp ",
        "group 1 pad [1, 1, 1, 1] stride [1, 1] dilation [1, 1]",
        "x1.barrier 0",
        "x1.barrier 1",
        "x1.store ",
    )
    for snippet in conv2d_expected_snippets:
        if snippet not in conv2d_asm:
            raise AssertionError(f"expected {snippet!r} in compiled conv2d IR:\n{conv2d_asm}")

    print(conv2d_asm)


if __name__ == "__main__":
    main()
