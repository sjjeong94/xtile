#!/usr/bin/env python3

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
        "x1.barrier {mode = 0}",
        "x1.barrier {mode = 1}",
        "x1.store ",
        "bank = 0",
        "space = 3",
        "shape = [8, 16]",
    )
    for snippet in expected_snippets:
        if snippet not in module_asm:
            raise AssertionError(f"expected {snippet!r} in compiled IR:\n{module_asm}")

    print(module_asm)


if __name__ == "__main__":
    main()
