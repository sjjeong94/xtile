#!/usr/bin/env python3

import xtile as xt

def main():
    module = xt.parse(
        """
        module {
          func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
            %0 = xt.add(%arg0, %arg0) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            func.return %0 : tensor<4xf32>
          }
        }
        """
    )
    lowered = xt.to_nova(module)
    if lowered is not module:
        raise AssertionError("xt.to_nova should return the original module object")

    module_asm = xt._module_asm(module)
    if "nova.elementwise" not in module_asm:
        raise AssertionError(f"expected nova.elementwise in lowered IR:\n{module_asm}")

    print(module_asm)


if __name__ == "__main__":
    main()
