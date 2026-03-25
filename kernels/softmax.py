import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import xtile as xt


@xt.kernel
def softmax_kernel(inp, out, trow, col):
    bid_x = xt.bid(0)

    x = xt.load(inp, index=(bid_x, 0), shape=(trow, col))
    x = x - xt.max(x, axis=-1)
    x = xt.exp(x)
    r = xt.sum(x, axis=-1)
    x = x / r
    xt.store(out, index=(bid_x, 0), tile=x)


def softmax(inp: xt.Array, out: xt.Array) -> object:
    assert inp.ndim == out.ndim == 2

    row, col = inp.shape

    trow = 64
    grid = (xt.cdiv(row, trow), 1, 1)

    ir = xt.convert(
        softmax_kernel,
        args=(inp, out, trow, col),
        grid=grid,
        double_buffering=True,
    )
    return ir


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", type=int, default=128)
    parser.add_argument("--col", type=int, default=128)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    row = args.row
    col = args.col
    inp = xt.Array(shape=(row, col), dtype=xt.float32)
    out = xt.Array(shape=(row, col), dtype=xt.float32)

    ir = softmax(inp, out)

    xt.save_compile_results(ir, f"compiled/softmax/softmax_{row}x{col}")


if __name__ == "__main__":
    main()
