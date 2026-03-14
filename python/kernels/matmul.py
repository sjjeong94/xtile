import xtile as xt


@xt.kernel
def matmul_kernel(
    a: xt.memref("?x128xf32"),
    b: xt.memref("128x?xf32"),
    result: xt.memref("?x?xf32"),
):
    tm, tn = 64, 64
    bid_x = xt.bid(0)
    bid_y = xt.bid(1)

    a_tile = xt.load(a, index=(bid_x, 0), shape=(tm, 128))
    b_tile = xt.load(b, index=(0, bid_y), shape=(128, tn), shared=1)
    result_tile = a_tile @ b_tile
    xt.store(result, index=(bid_x, bid_y), tile=result_tile)
