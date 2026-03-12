import xtile as xt


@xt.kernel
def add_kernel(
    a: xt.memref("?xf32"),
    b: xt.memref("?xf32"),
    result: xt.memref("?xf32"),
):
    tile_size = 16
    block_id = xt.bid(0)

    a_tile = xt.load(a, index=(block_id,), shape=(tile_size,))
    b_tile = xt.load(b, index=(block_id,), shape=(tile_size,))
    result_tile = a_tile + b_tile
    xt.store(result, index=(block_id,), tile=result_tile)


def main() -> None:
    module = xt.convert(add_kernel)
    print(xt.dump(module))


if __name__ == "__main__":
    main()
