#!/usr/bin/env python3

import xtile as xt


def main():
    module = xt.parse(
        """
        module {
          func.func @insert_barriers(%src: memref<16x16xf32>, %dst: memref<16x16xf32>) {
            %0 = nova.load(%src) {start = [0, 0]} : memref<16x16xf32> -> tensor<16x16xf32>
            %1 = nova.square(%0) : tensor<16x16xf32> -> tensor<16x16xf32>
            nova.store(%1, %dst) {start = [0, 0]} : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
            func.return
          }
        }
        """
    )

    barriered = xt.nova_barrier(module)
    if barriered is not module:
        raise AssertionError("xt.nova_barrier should return the original module object")

    module_asm = xt._module_asm(module)
    if "nova.barrier() {mode = 0 : i32}" not in module_asm:
        raise AssertionError(f"expected compute barrier in IR:\n{module_asm}")
    if "nova.barrier() {mode = 1 : i32}" not in module_asm:
        raise AssertionError(f"expected return barrier in IR:\n{module_asm}")

    print(module_asm)


if __name__ == "__main__":
    main()
