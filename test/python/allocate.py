#!/usr/bin/env python3

import xtile as xt

def main():
    module = xt.parse(
        """
        module {
          func.func @allocate_basic(%src: memref<64x16xf32>, %dst: memref<64x16xf32>) {
            %0 = nova.load(%src) {start = [0, 0]} : memref<64x16xf32> -> tensor<16x16xf32>
            %1 = nova.square(%0) : tensor<16x16xf32> -> tensor<16x16xf32>
            nova.store(%1, %dst) {start = [0, 0]} : (tensor<16x16xf32>, memref<64x16xf32>) -> ()
            nova.free(%1) : tensor<16x16xf32>
            func.return
          }
        }
        """
    )

    allocated = xt.nova_allocate(module)
    if allocated is not module:
        raise AssertionError("xt.nova_allocate should return the original module object")

    module_asm = xt._module_asm(module)
    if "bank0 = 0 : i64" not in module_asm:
        raise AssertionError(f"expected allocated bank annotation in IR:\n{module_asm}")
    if "space = 3 : i64" not in module_asm:
        raise AssertionError(f"expected allocated space annotation in IR:\n{module_asm}")
    if "nova.free(" in module_asm:
        raise AssertionError(f"expected nova.free ops to be removed:\n{module_asm}")

    print(module_asm)


if __name__ == "__main__":
    main()
