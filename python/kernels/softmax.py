import xtile as xt


@xt.kernel
def softmax_kernel(
    a: xt.memref("?x?xf32"),
    result: xt.memref("?x?xf32"),
):
    block_id = xt.bid(0)
    a_tile = xt.load(a, index=(block_id, 0), shape=(16, 16))
    row_max = xt.max(a_tile)
    shifted = a_tile - row_max
    exp_shifted = xt.exp(shifted)
    row_sum = xt.sum(exp_shifted)
    inv_row_sum = xt.reciprocal(row_sum)
    softmax_tile = exp_shifted * inv_row_sum
    xt.store(result, index=(block_id, 0), tile=softmax_tile)
