#!/usr/bin/env python3

import xtile as xt


def check_rowwise_softmax():
    a = xt.Array(shape=(128, 64), dtype=xt.float32)
    b = xt.Array(shape=(128, 64), dtype=xt.float32)

    @xt.kernel
    def rowwise_softmax(x, y):
        bid_x = xt.bid(0)

        t = xt.load(x, index=(bid_x, 0), shape=(64, 64))
        r = xt.max(t)
        t = t - r
        t = xt.exp(t)
        r = xt.sum(t)
        r = xt.reciprocal(r)
        t = t * r
        xt.store(y, index=(bid_x, 0), tile=t)

    module = xt.convert(rowwise_softmax, args=(a, b), grid=(2, 1, 1), double_buffering=True)

    expected = """
func.func @rowwise_softmax(%input: memref<128x64xf32>, %output: memref<128x64xf32>) attributes {xt.grid = array<i32: 2, 1, 1>, xt.double_buffering = 1 : i32} {
  %bid:3 = xt.get_tile_block_id : i32, i32, i32
  %zero = arith.constant 0 : i32

  %x = xt.load(%input, %bid#0, %zero) : (memref<128x64xf32>, i32, i32) -> tensor<64x64xf32>
  %row_max = xt.reduce_max(%x) : tensor<64x64xf32> -> tensor<64x1xf32>
  %shifted = xt.sub(%x, %row_max) : tensor<64x64xf32>, tensor<64x1xf32> -> tensor<64x64xf32>
  %exp = xt.exp(%shifted) : tensor<64x64xf32> -> tensor<64x64xf32>
  %row_sum = xt.reduce_sum(%exp) : tensor<64x64xf32> -> tensor<64x1xf32>
  %inv_sum = xt.reciprocal(%row_sum) : tensor<64x1xf32> -> tensor<64x1xf32>
  %normalized = xt.mul(%exp, %inv_sum) : tensor<64x64xf32>, tensor<64x1xf32> -> tensor<64x64xf32>
  xt.store(%normalized, %output, %bid#0, %zero) : (tensor<64x64xf32>, memref<128x64xf32>, i32, i32) -> ()
  func.return
}
""".strip()
    actual = str(module).strip()
    if actual != expected:
        raise AssertionError(f"unexpected MLIR output:\n{actual}\n!=\n{expected}")

    serialized = xt.serialize(module)
    if serialized is not module:
        raise AssertionError("xt.serialize should preserve the DSL module wrapper")

    return actual


def check_matmul_pipeline():
    a = xt.Array(shape=(1024, 256), dtype=xt.int8)
    b = xt.Array(shape=(256, 512), dtype=xt.int8)
    c = xt.Array(shape=(1024, 512), dtype=xt.int8)
    s = xt.Array(shape=(1, 512), dtype=xt.float32)
    bias = xt.Array(shape=(1, 512), dtype=xt.float32)

    tile_m = 64
    tile_n = 64
    tile_k = 256

    @xt.kernel
    def matmul(lhs, rhs, res, scale, bias, tm, tn, tk):
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

    module = xt.convert(
        matmul,
        args=(a, b, c, s, bias, tile_m, tile_n, tile_k),
        grid=(16, 8, 1),
        double_buffering=True,
    )
    actual = str(module)

    expected_snippets = (
        "func.func @matmul(",
        "xt.grid = array<i32: 16, 8, 1>",
        "xt.double_buffering = 1 : i32",
        "xt.load(",
        "{shared = 1 : i64}",
        "xt.matmul(",
        "xt.mul(",
        "xt.add(",
        "xt.ftoi(",
        "xt.store(",
        "tensor<64x256xi8>",
        "tensor<256x64xi8>",
        "tensor<64x64xf32>",
        "tensor<64x64xi8>",
    )
    for snippet in expected_snippets:
        if snippet not in actual:
            raise AssertionError(f"expected {snippet!r} in emitted MLIR:\n{actual}")

    return actual


def main():
    softmax = check_rowwise_softmax()
    matmul = check_matmul_pipeline()
    print(softmax)
    print()
    print(matmul)


if __name__ == "__main__":
    main()
