#!/usr/bin/env python3

from pathlib import Path

import xtile as xt


def main():
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

    expected = Path("test.mlir").read_text(encoding="utf-8").strip()
    actual = str(module).strip()
    if actual != expected:
        raise AssertionError(f"unexpected MLIR output:\n{actual}\n!=\n{expected}")

    serialized = xt.serialize(module)
    if serialized is not module:
        raise AssertionError("xt.serialize should preserve the DSL module wrapper")

    print(actual)


if __name__ == "__main__":
    main()
