#!/usr/bin/env python3

import xtile as xt

def main():
    module = xt.parse(
        """
        module {
          func.func @fuse_scalar_mul_then_add(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32> {
            %0 = nova.scalar(%arg0) {mode = 2 : i32, rhs = 3.000000e+00 : f32} : tensor<16x16xf32> -> tensor<16x16xf32>
            %1 = nova.scalar(%0) {mode = 1 : i32, rhs = 4.000000e+00 : f32} : tensor<16x16xf32> -> tensor<16x16xf32>
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
    if module_asm.count("nova.scalar(") != 0:
        raise AssertionError(f"expected scalar chain to be folded away:\n{module_asm}")

    print(module_asm)


if __name__ == "__main__":
    main()
