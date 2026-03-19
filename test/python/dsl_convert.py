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


def check_missing_high_level_ops():
    unary_in = xt.Array(shape=(8, 8), dtype=xt.float32)
    unary_out = xt.Array(shape=(8, 8), dtype=xt.float32)
    reshape_out = xt.Array(shape=(4, 16), dtype=xt.float32)
    transpose_in = xt.Array(shape=(2, 3, 4), dtype=xt.float32)
    transpose_out = xt.Array(shape=(2, 4, 3), dtype=xt.float32)

    @xt.kernel
    def unary_and_shape(x, y, z, t_in, t_out):
        tile = xt.load(x, index=(0, 0), shape=(8, 8))
        tile = xt.cos(tile)
        tile = xt.sin(tile)
        tile = xt.rsqrt(tile)
        tile = xt.sigmoid(tile)
        tile = xt.tanh(tile)
        tile = xt.silu(tile)
        xt.store(y, index=(0, 0), tile=tile)

        reshaped = xt.reshape(tile, shape=(4, 16))
        xt.store(z, index=(0, 0), tile=reshaped)

        t3 = xt.load(t_in, index=(0, 0, 0), shape=(2, 3, 4))
        t3 = xt.transpose(t3)
        xt.store(t_out, index=(0, 0, 0), tile=t3)

    unary_module = xt.convert(
        unary_and_shape,
        args=(unary_in, unary_out, reshape_out, transpose_in, transpose_out),
        grid=(1, 1, 1),
        double_buffering=False,
    )
    unary_actual = str(unary_module)
    unary_expected = (
        "xt.cos(",
        "xt.sin(",
        "xt.rsqrt(",
        "xt.sigmoid(",
        "xt.tanh(",
        "xt.silu(",
        "xt.reshape(",
        "tensor<4x16xf32>",
        "xt.transpose(",
        "tensor<2x4x3xf32>",
    )
    for snippet in unary_expected:
        if snippet not in unary_actual:
            raise AssertionError(f"expected {snippet!r} in unary/shape MLIR:\n{unary_actual}")

    lhs = xt.Array(shape=(16, 32), dtype=xt.int8)
    rhs = xt.Array(shape=(32, 8), dtype=xt.int8)
    acc_in = xt.Array(shape=(16, 8), dtype=xt.float32)
    acc_out = xt.Array(shape=(16, 8), dtype=xt.float32)
    conv_in = xt.Array(shape=(1, 5, 5, 4), dtype=xt.int8)
    conv_filter = xt.Array(shape=(3, 3, 4, 6), dtype=xt.int8)
    conv_out = xt.Array(shape=(1, 3, 3, 6), dtype=xt.float32)
    depth_filter = xt.Array(shape=(3, 3, 1, 4), dtype=xt.int8)
    depth_out = xt.Array(shape=(1, 3, 3, 4), dtype=xt.float32)

    @xt.kernel
    def compute_ops(a, b, acc_src, acc_dst, cin, cfilter, cout, dfilter, dout):
        a_tile = xt.load(a, index=(0, 0), shape=(16, 32))
        b_tile = xt.load(b, index=(0, 0), shape=(32, 8))
        acc_tile = xt.load(acc_src, index=(0, 0), shape=(16, 8))
        acc_tile = xt.mma(a_tile, b_tile, acc_tile)
        xt.store(acc_dst, index=(0, 0), tile=acc_tile)

        cin_tile = xt.load(cin, index=(0, 0, 0, 0), shape=(1, 5, 5, 4))
        cfilter_tile = xt.load(cfilter, index=(0, 0, 0, 0), shape=(3, 3, 4, 6))
        conv_tile = xt.conv2d(
            cin_tile,
            cfilter_tile,
            pad=(0, 0, 0, 0),
            stride=(1, 1),
            dilation=(1, 1),
        )
        xt.store(cout, index=(0, 0, 0, 0), tile=conv_tile)

        dfilter_tile = xt.load(dfilter, index=(0, 0, 0, 0), shape=(3, 3, 1, 4))
        depth_tile = xt.depthwise_conv2d(
            cin_tile,
            dfilter_tile,
            pad=(0, 0, 0, 0),
            stride=(1, 1),
            dilation=(1, 1),
        )
        xt.store(dout, index=(0, 0, 0, 0), tile=depth_tile)

    compute_module = xt.convert(
        compute_ops,
        args=(lhs, rhs, acc_in, acc_out, conv_in, conv_filter, conv_out, depth_filter, depth_out),
        grid=(1, 1, 1),
        double_buffering=False,
    )
    compute_actual = str(compute_module)
    compute_expected = (
        "xt.mma(",
        "tensor<16x8xf32>",
        "xt.conv2d(",
        "pad = array<i64: 0, 0, 0, 0>",
        "stride = array<i64: 1, 1>",
        "dilation = array<i64: 1, 1>",
        "tensor<1x3x3x6xf32>",
        "xt.depthwise_conv2d(",
        "tensor<1x3x3x4xf32>",
    )
    for snippet in compute_expected:
        if snippet not in compute_actual:
            raise AssertionError(f"expected {snippet!r} in compute MLIR:\n{compute_actual}")

    return unary_actual, compute_actual


def main():
    softmax = check_rowwise_softmax()
    matmul = check_matmul_pipeline()
    unary_shape, compute = check_missing_high_level_ops()
    print(softmax)
    print()
    print(matmul)
    print()
    print(unary_shape)
    print()
    print(compute)


if __name__ == "__main__":
    main()
