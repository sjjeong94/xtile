#!/usr/bin/env python3

import xtile as xt

def main():
    module = xt.parse(
        """
        module {
          func.func @fuse_scalar_mul_then_add(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32> {
            %0 = nova.scalar 2 %arg0, 3.000000e+00 : tensor<16x16xf32> -> tensor<16x16xf32>
            %1 = nova.scalar 1 %0, 4.000000e+00 : tensor<16x16xf32> -> tensor<16x16xf32>
            func.return %1 : tensor<16x16xf32>
          }
        }
        """
    )

    optimized = xt.nova_optimize(module)
    if optimized is not module:
        raise AssertionError("xt.nova_optimize should return the original module object")

    module_asm = xt._module_asm(module)
    if "nova.scalar_fma" not in module_asm:
        raise AssertionError(f"expected nova.scalar_fma in optimized IR:\n{module_asm}")
    if "nova.scalar " in module_asm:
        raise AssertionError(f"expected scalar chain to be folded away:\n{module_asm}")

    print(module_asm)


if __name__ == "__main__":
    main()
