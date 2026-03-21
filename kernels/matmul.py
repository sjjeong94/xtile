import argparse

import xtile as xt


@xt.kernel
def matmul_kernel(lhs, rhs, res, scale, bias, tm, tn, tk):
    bid_x = xt.bid(0)
    bid_y = xt.bid(1)

    lhs_tile = xt.load(lhs, index=(bid_x, 0), shape=(tm, tk))
    rhs_tile = xt.load(rhs, index=(0, bid_y), shape=(tk, tn), shared=1)
    s_tile = xt.load(scale, index=(0, bid_y), shape=(1, tn), shared=1)
    b_tile = xt.load(bias, index=(0, bid_y), shape=(1, tn), shared=1)

    t = xt.matmul(lhs_tile, rhs_tile)
    t = t * s_tile + b_tile
    t = xt.astype(t, dtype=xt.int8)
    xt.store(res, index=(bid_x, bid_y), tile=t)


def matmul(m: int, n: int, k: int) -> object:
    a = xt.Array(shape=(m, k), dtype=xt.int8)
    b = xt.Array(shape=(k, n), dtype=xt.int8)
    c = xt.Array(shape=(m, n), dtype=xt.int8)
    s = xt.Array(shape=(1, n), dtype=xt.float32)
    bias = xt.Array(shape=(1, n), dtype=xt.float32)

    tm = 64
    tn = 64

    ir = xt.convert(
        matmul_kernel,
        args=(a, b, c, s, bias, tm, tn, k),
        grid=(xt.cdiv(m, tm), xt.cdiv(n, tn), 1),
        double_buffering=True,
    )
    return ir


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=256)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    m = args.m
    n = args.n
    k = args.k

    ir = matmul(m, n, k)

    xt.save_compile_results(ir, f"compiled/matmul/matmul_{m}x{n}x{k}")


if __name__ == "__main__":
    main()
