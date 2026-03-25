import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import xtile as xt


@xt.kernel
def conv2d_kernel(inp, filt, out, tile_h, tile_w, cin, cout, kh, kw):
    bid_n = xt.bid(0)
    bid_h = xt.bid(1)
    bid_w = xt.bid(2)
    zero = 0

    filt_tile = xt.load(
        filt,
        index=(zero, zero, zero, zero),
        shape=(kh, kw, cin, cout),
        shared=2,
    )
    out_tile = xt.load_conv2d(
        inp,
        filt_tile,
        index=(bid_n, bid_h, bid_w, zero),
        shape=(1, tile_h, tile_w, cout),
        group=1,
        pad=(kh // 2, kw // 2, kh // 2, kw // 2),
        stride=(1, 1),
        dilation=(1, 1),
    )
    xt.store(out, index=(bid_n, bid_h, bid_w, zero), tile=out_tile)


def conv2d(n: int, h: int, w: int, cin: int, cout: int, kh: int, kw: int) -> object:
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("conv2d kernel currently requires odd kh and kw")
    if n <= 0 or h <= 0 or w <= 0 or cin <= 0 or cout <= 0:
        raise ValueError("all conv2d dimensions must be positive")

    tile_h = 32
    tile_w = 64
    if h % tile_h != 0:
        raise ValueError(f"h must be divisible by {tile_h}")
    if w % tile_w != 0:
        raise ValueError(f"w must be divisible by {tile_w}")

    inp = xt.Array(shape=(n, h, w, cin), dtype=xt.int8)
    filt = xt.Array(shape=(kh, kw, cin, cout), dtype=xt.int8)
    out = xt.Array(shape=(n, h, w, cout), dtype=xt.float32)

    ir = xt.convert(
        conv2d_kernel,
        args=(inp, filt, out, tile_h, tile_w, cin, cout, kh, kw),
        grid=(n, xt.cdiv(h, tile_h), xt.cdiv(w, tile_w)),
        double_buffering=True,
    )
    return ir


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--h", type=int, default=32)
    parser.add_argument("--w", type=int, default=64)
    parser.add_argument("--cin", type=int, default=128)
    parser.add_argument("--cout", type=int, default=64)
    parser.add_argument("--kh", type=int, default=3)
    parser.add_argument("--kw", type=int, default=3)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ir = conv2d(args.n, args.h, args.w, args.cin, args.cout, args.kh, args.kw)
    save_dir = (
        f"compiled/conv2d/conv2d_{args.n}x{args.h}x{args.w}x"
        f"{args.cin}x{args.cout}x{args.kh}x{args.kw}"
    )

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    xt.save_ir(ir, f"{save_dir}/0_xt.mlir")
    ir = xt.xt_serialize(ir)
    xt.save_ir(ir, f"{save_dir}/1_xt_serialize.mlir")
    ir = xt.xt_to_nova(ir)
    xt.save_ir(ir, f"{save_dir}/2_xt_to_nova.mlir")
    ir = xt.nova_optimize(ir)
    xt.save_ir(ir, f"{save_dir}/3_nova_optimize.mlir")
    ir = xt.nova_threading(ir)
    xt.save_ir(ir, f"{save_dir}/4_nova_threading.mlir")
    ir = xt.nova_barrier(ir)
    xt.save_ir(ir, f"{save_dir}/5_nova_barrier.mlir")
    ir = xt.nova_allocate(ir)
    xt.save_ir(ir, f"{save_dir}/6_nova_allocate.mlir")
    print("compiled kernel ->", save_dir)
    print("skipped 7_nova_to_x1.mlir because nova.conv2d lowering is not implemented")


if __name__ == "__main__":
    main()
