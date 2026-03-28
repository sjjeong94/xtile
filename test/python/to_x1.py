#!/usr/bin/env python3

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) in sys.path:
    sys.path.remove(str(THIS_DIR))

import xtile as xt


def main():
    barrier_module = xt.parse(
        """
        module {
          func.func @barrier_only() {
            nova.barrier 1
            func.return
          }
        }
        """
    )

    lowered = xt.nova_to_x1(barrier_module)
    if lowered is not barrier_module:
        raise AssertionError("xt.nova_to_x1 should return the original module object")

    barrier_asm = xt._module_asm(barrier_module)
    if "x1.barrier 1" not in barrier_asm:
        raise AssertionError(f"expected x1.barrier in lowered IR:\n{barrier_asm}")
    if "nova.barrier " in barrier_asm:
        raise AssertionError(f"expected nova.barrier to be rewritten away:\n{barrier_asm}")

    conv2d_module = xt.parse(
        """
        module {
          func.func @conv2d_only(%src: memref<1x32x64x128xi8>, %filter: memref<3x3x128x64xi8>, %dst: memref<1x32x64x64xf32>) {
            %0 = nova.load %src [0, 0, 0, 0] : memref<1x32x64x128xi8> -> tensor<1x32x64x128xi8, #nova.layout<range0 [0, 0, 0, 0] [1, 32, 64, 128], bank0 = 0, space = 3>>
            %1 = nova.load %filter [0, 0, 0, 0] : memref<3x3x128x64xi8> -> tensor<3x3x128x64xi8, #nova.layout<range0 [0, 0, 0, 0] [3, 3, 128, 64], bank0 = 1, space = 3>>
            %2 = nova.conv2d %0, %1 group 1 pad [1, 1, 1, 1] stride [1, 1] dilation [1, 1] : tensor<1x32x64x128xi8, #nova.layout<range0 [0, 0, 0, 0] [1, 32, 64, 128], bank0 = 0, space = 3>>, tensor<3x3x128x64xi8, #nova.layout<range0 [0, 0, 0, 0] [3, 3, 128, 64], bank0 = 1, space = 3>> -> tensor<1x32x64x64xf32, #nova.layout<range0 [0, 0, 0, 0] [1, 32, 64, 64], bank0 = 2, space = 3>>
            nova.store %2, %dst [0, 0, 0, 0] : (tensor<1x32x64x64xf32, #nova.layout<range0 [0, 0, 0, 0] [1, 32, 64, 64], bank0 = 2, space = 3>>, memref<1x32x64x64xf32>) -> ()
            func.return
          }
        }
        """
    )

    lowered = xt.nova_to_x1(conv2d_module)
    if lowered is not conv2d_module:
        raise AssertionError("xt.nova_to_x1 should return the original module object")

    conv2d_asm = xt._module_asm(conv2d_module)
    if "x1.conv2d inp 0 filter 1 out 2" not in conv2d_asm:
        raise AssertionError(f"expected x1.conv2d in lowered IR:\n{conv2d_asm}")
    if "nova.conv2d " in conv2d_asm:
        raise AssertionError(f"expected nova.conv2d to be rewritten away:\n{conv2d_asm}")

    print(barrier_asm)
    print(conv2d_asm)


if __name__ == "__main__":
    main()
