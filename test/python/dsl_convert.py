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
  %row_max = xt.reduce_max(%x) {axis = 1 : i64} : tensor<64x64xf32> -> tensor<64x1xf32>
  %shifted = xt.sub(%x, %row_max) : tensor<64x64xf32>, tensor<64x1xf32> -> tensor<64x64xf32>
  %exp = xt.exp(%shifted) : tensor<64x64xf32> -> tensor<64x64xf32>
  %row_sum = xt.reduce_sum(%exp) {axis = 1 : i64} : tensor<64x64xf32> -> tensor<64x1xf32>
  %inv_sum = xt.reciprocal(%row_sum) : tensor<64x1xf32> -> tensor<64x1xf32>
  %normalized = xt.mul(%exp, %inv_sum) : tensor<64x64xf32>, tensor<64x1xf32> -> tensor<64x64xf32>
  xt.store(%normalized, %output, %bid#0, %zero) : (tensor<64x64xf32>, memref<128x64xf32>, i32, i32) -> ()
  func.return
}
""".strip()
    actual = str(module).strip()
    if actual != expected:
        raise AssertionError(f"unexpected MLIR output:\n{actual}\n!=\n{expected}")

    serialized = xt.xt_serialize(module)
    if serialized is not module:
        raise AssertionError("xt.xt_serialize should preserve the DSL module wrapper")

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


def check_truediv_lowers_to_mul_reciprocal():
    a = xt.Array(shape=(8, 8), dtype=xt.float32)
    b = xt.Array(shape=(8, 1), dtype=xt.float32)
    out = xt.Array(shape=(8, 8), dtype=xt.float32)

    @xt.kernel
    def divide_kernel(x, y, z):
        tile = xt.load(x, index=(0, 0), shape=(8, 8))
        denom = xt.load(y, index=(0, 0), shape=(8, 1))
        tile = tile / denom
        xt.store(z, index=(0, 0), tile=tile)

    module = xt.convert(
        divide_kernel,
        args=(a, b, out),
        grid=(1, 1, 1),
        double_buffering=False,
    )
    actual = str(module)
    expected_snippets = (
        "xt.reciprocal(",
        "xt.mul(",
        "tensor<8x1xf32>",
        "tensor<8x8xf32>",
    )
    for snippet in expected_snippets:
        if snippet not in actual:
            raise AssertionError(f"expected {snippet!r} in truediv MLIR:\n{actual}")
    if "xt.div(" in actual:
        raise AssertionError(f"did not expect a dedicated division op:\n{actual}")

    return actual


def check_load_conv2d_emits_xt_op():
    src = xt.Array(shape=(1, 34, 66, 128), dtype=xt.int8)
    filt_src = xt.Array(shape=(3, 3, 128, 64), dtype=xt.int8)
    out = xt.Array(shape=(1, 32, 64, 32), dtype=xt.float32)

    @xt.kernel
    def load_conv2d_kernel(x, f, y):
        filt = xt.load(f, index=(0, 0, 0, 0), shape=(3, 3, 128, 64))
        tile = xt.load_conv2d(
            x,
            filt,
            index=(0, 0, 0, 0),
            shape=(1, 32, 64, 32),
            group=1,
            pad=(1, 1, 1, 1),
            stride=(1, 1),
            dilation=(1, 1),
        )
        xt.store(y, index=(0, 0, 0, 0), tile=tile)

    module = xt.convert(
        load_conv2d_kernel,
        args=(src, filt_src, out),
        grid=(1, 1, 1),
        double_buffering=False,
    )
    actual = str(module)
    expected_snippets = (
        "xt.load_conv2d(",
        "tensor<3x3x128x64xi8>",
        "tensor<1x32x64x32xf32>",
        "group = 1 : i64",
        "pad = array<i64: 1, 1, 1, 1>",
    )
    for snippet in expected_snippets:
        if snippet not in actual:
            raise AssertionError(f"expected {snippet!r} in load_conv2d MLIR:\n{actual}")

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
    @xt.kernel
    def compute_ops(a, b, acc_src, acc_dst):
        a_tile = xt.load(a, index=(0, 0), shape=(16, 32))
        b_tile = xt.load(b, index=(0, 0), shape=(32, 8))
        acc_tile = xt.load(acc_src, index=(0, 0), shape=(16, 8))
        acc_tile = xt.mma(a_tile, b_tile, acc_tile)
        xt.store(acc_dst, index=(0, 0), tile=acc_tile)

    compute_module = xt.convert(
        compute_ops,
        args=(lhs, rhs, acc_in, acc_out),
        grid=(1, 1, 1),
        double_buffering=False,
    )
    compute_actual = str(compute_module)
    compute_expected = (
        "xt.mma(",
        "tensor<16x8xf32>",
    )
    for snippet in compute_expected:
        if snippet not in compute_actual:
            raise AssertionError(f"expected {snippet!r} in compute MLIR:\n{compute_actual}")

    reduce_in = xt.Array(shape=(8, 8), dtype=xt.float32)
    reduce_out = xt.Array(shape=(1, 8), dtype=xt.float32)

    @xt.kernel
    def column_reduce(x, y):
        tile = xt.load(x, index=(0, 0), shape=(8, 8))
        tile = xt.sum(tile, axis=0)
        xt.store(y, index=(0, 0), tile=tile)

    reduce_module = xt.convert(
        column_reduce,
        args=(reduce_in, reduce_out),
        grid=(1, 1, 1),
        double_buffering=False,
    )
    reduce_actual = str(reduce_module)
    reduce_expected = (
        "xt.reduce_sum(",
        "{axis = 0 : i64}",
        "tensor<1x8xf32>",
    )
    for snippet in reduce_expected:
        if snippet not in reduce_actual:
            raise AssertionError(f"expected {snippet!r} in axis-0 reduce MLIR:\n{reduce_actual}")

    return unary_actual, compute_actual, reduce_actual


def main():
    softmax = check_rowwise_softmax()
    matmul = check_matmul_pipeline()
    truediv = check_truediv_lowers_to_mul_reciprocal()
    load_conv2d = check_load_conv2d_emits_xt_op()
    unary_shape, compute, reduce_axis0 = check_missing_high_level_ops()
    print(softmax)
    print()
    print(matmul)
    print()
    print(truediv)
    print()
    print(load_conv2d)
    print()
    print(unary_shape)
    print()
    print(compute)
    print()
    print(reduce_axis0)


if __name__ == "__main__":
    main()
