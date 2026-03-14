import xtile as xt


@xt.kernel
def softmax_kernel(
    a: xt.memref("?xf32"),
    result: xt.memref("?xf32"),
):
    block_id = xt.bid(0)
    a_tile = xt.load(a, index=(block_id,), shape=(16, 16))
    row_max = xt.max(a_tile)
    shifted = a_tile - row_max
    exp_shifted = xt.exp(shifted)
    row_sum = xt.sum(exp_shifted)
    inv_row_sum = xt.reciprocal(row_sum)
    softmax_tile = exp_shifted * inv_row_sum
    xt.store(result, index=(block_id,), tile=softmax_tile)
