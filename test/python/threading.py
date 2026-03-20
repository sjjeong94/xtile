#!/usr/bin/env python3

import xtile as xt


def main():
    module = xt.parse(
        """
        module {
          func.func @load_sets_threading(%src: memref<10x8xf32>) {
            %0 = nova.load(%src) {start = array<i64: 0, 0>} : memref<10x8xf32> -> tensor<5x8xf32>
            nova.free(%0) : tensor<5x8xf32>
            func.return
          }
        }
        """
    )

    threaded = xt.nova_threading(module)
    if threaded is not module:
        raise AssertionError("xt.nova_threading should return the original module object")

    module_asm = xt._module_asm(module)
    if "threading = 3 : i64" not in module_asm:
        raise AssertionError(f"expected threading annotation in IR:\n{module_asm}")

    print(module_asm)


if __name__ == "__main__":
    main()
