import argparse

import xtile as xt


@xt.kernel
def add_kernel(lhs, rhs, out, trow, col):
    bid_x = xt.bid(0)

    a = xt.load(lhs, index=(bid_x, 0), shape=(trow, col))
    b = xt.load(rhs, index=(bid_x, 0), shape=(trow, col))
    c = a + b
    xt.store(out, index=(bid_x, 0), tile=c)


def add(lhs: xt.Array, rhs: xt.Array, out: xt.Array) -> object:
    assert lhs.ndim == rhs.ndim == out.ndim == 2

    row, col = lhs.shape

    trow = 64
    grid = (xt.cdiv(row, trow), 1, 1)

    ir = xt.convert(
        add_kernel,
        args=(lhs, rhs, out, trow, col),
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
    lhs = xt.Array(shape=(row, col), dtype=xt.float32)
    rhs = xt.Array(shape=(row, col), dtype=xt.float32)
    out = xt.Array(shape=(row, col), dtype=xt.float32)

    ir = add(lhs, rhs, out)

    xt.save_compile_results(ir, f"compiled/add/add_{row}x{col}")


if __name__ == "__main__":
    main()
