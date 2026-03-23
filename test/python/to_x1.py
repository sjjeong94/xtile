#!/usr/bin/env python3

import xtile as xt


def main():
    module = xt.parse(
        """
        module {
          func.func @barrier_only() {
            nova.barrier 1
            func.return
          }
        }
        """
    )

    lowered = xt.nova_to_x1(module)
    if lowered is not module:
        raise AssertionError("xt.nova_to_x1 should return the original module object")

    module_asm = xt._module_asm(module)
    if "x1.barrier {mode = 1}" not in module_asm:
        raise AssertionError(f"expected x1.barrier in lowered IR:\n{module_asm}")
    if "nova.barrier " in module_asm:
        raise AssertionError(f"expected nova.barrier to be rewritten away:\n{module_asm}")

    print(module_asm)


if __name__ == "__main__":
    main()
