#!/usr/bin/env python3

import xtile as xt


def main():
    module = xt.parse(
        """
        module {
          func.func @load_sets_threading(%src: memref<10x8xf32>) {
            %0 = nova.load %src [0, 0] : memref<10x8xf32> -> tensor<5x8xf32>
            func.return
          }
        }
        """
    )

    threaded = xt.nova_threading(module)
    if threaded is not module:
        raise AssertionError("xt.nova_threading should return the original module object")

    module_asm = xt._module_asm(module)
    if "shape0 = array<i64: 3, 8>" not in module_asm or "shape1 = array<i64: 2, 8>" not in module_asm:
        raise AssertionError(f"expected slice annotations in IR:\n{module_asm}")
    if "threading =" in module_asm:
        raise AssertionError(f"did not expect threading annotation in IR:\n{module_asm}")

    print(module_asm)


if __name__ == "__main__":
    main()
