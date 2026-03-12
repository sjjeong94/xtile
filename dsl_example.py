import sys

MLIR_PYTHON_ROOT = (
    "/home/sjjeong94/projects/llvm-project/build/tools/mlir/python_packages/mlir_core"
)
if MLIR_PYTHON_ROOT not in sys.path:
    sys.path.insert(0, MLIR_PYTHON_ROOT)

PYTHON_ROOT = "/home/sjjeong94/projects/xt/.worktrees/python-dsl/python"
if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)

import xtile as xt


@xt.kernel
def add_kernel(
    a: xt.memref("?xf32"),
    b: xt.memref("?xf32"),
    result: xt.memref("?xf32"),
):
    tile_size = 16
    block_id = xt.bid(0)

    a_tile = xt.load(a, index=(block_id,), shape=(tile_size,))
    b_tile = xt.load(b, index=(block_id,), shape=(tile_size,))
    result_tile = a_tile + b_tile
    xt.store(result, index=(block_id,), tile=result_tile)


def main() -> None:
    module = xt.convert(add_kernel)
    print(xt.dump(module))


if __name__ == "__main__":
    main()
