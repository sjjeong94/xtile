import xtile as xt


@xt.kernel
def layernorm_kernel(
    a: xt.memref("?xf32"),
    result: xt.memref("?xf32"),
):
    block_id = xt.bid(0)
    a_tile = xt.load(a, index=(block_id,), shape=(16, 16))
    inv_cols = xt.full(shape=(1, 1), value=0.0625)
    row_sum = xt.sum(a_tile)
    mean = row_sum * inv_cols
    centered = a_tile - mean
    sq = centered * centered
    var_sum = xt.sum(sq)
    var = var_sum * inv_cols
    inv_std = xt.rsqrt(var)
    normalized = centered * inv_std
    xt.store(result, index=(block_id,), tile=normalized)
