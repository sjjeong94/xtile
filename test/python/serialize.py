#!/usr/bin/env python3

import xtile as xt


def main():
    module = xt.parse(
        """
        module {
          func.func @serialize_grid(%src: memref<128x16xf32>, %dst: memref<128x16xf32>) attributes {xt.grid = array<i32: 2, 1, 1>} {
            %bid:3 = xt.get_tile_block_id : i32, i32, i32
            %0 = xt.load(%src, %bid#0, %bid#1) : (memref<128x16xf32>, i32, i32) -> tensor<16x16xf32>
            xt.store(%0, %dst, %bid#0, %bid#1) : (tensor<16x16xf32>, memref<128x16xf32>, i32, i32) -> ()
            func.return
          }
        }
        """
    )

    serialized = xt.serialize(module)
    if serialized is not module:
        raise AssertionError("xt.serialize should return the original module object")

    module_asm = xt._module_asm(module)
    if "xt.get_tile_block_id" in module_asm:
        raise AssertionError(
            f"expected xt.get_tile_block_id to be eliminated:\n{module_asm}"
        )
    if module_asm.count("arith.constant 0 : i32") < 2:
        raise AssertionError(f"expected serialized grid constants in IR:\n{module_asm}")

    print(module_asm)


if __name__ == "__main__":
    main()
