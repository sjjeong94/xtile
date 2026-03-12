import xtile as xt

from mlir import ir
import pytest

from xtile.errors import XTConversionError


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


def test_convert_add_kernel_to_mlir_module():
    module = xt.convert(add_kernel)

    assert isinstance(module, ir.Module)

    dumped = xt.dump(module)
    assert "func.func @add_kernel" in dumped
    assert "xt.get_tile_block_id" in dumped
    assert dumped.count("xt.load") == 2
    assert "xt.add" in dumped
    assert "xt.store" in dumped


@xt.kernel
def exp_kernel(
    a: xt.memref("?xf32"),
    result: xt.memref("?xf32"),
):
    tile_size = 16
    block_id = xt.bid(0)
    a_tile = xt.load(a, index=(block_id,), shape=(tile_size,))
    result_tile = xt.exp(a_tile)
    xt.store(result, index=(block_id,), tile=result_tile)


def test_convert_unary_kernel():
    module = xt.convert(exp_kernel)
    dumped = xt.dump(module)

    assert "func.func @exp_kernel" in dumped
    assert "xt.exp" in dumped
    assert "tensor<16xf32>" in dumped


@xt.kernel
def reshape_transpose_kernel(
    a: xt.memref("?xf32"),
    result: xt.memref("?xf32"),
):
    block_id = xt.bid(0)
    tile = xt.load(a, index=(block_id,), shape=(64, 16))
    reshaped = xt.reshape(tile, shape=(2, 32, 16))
    transposed = xt.transpose(reshaped, order=(0, 2, 1))
    flattened = xt.reshape(transposed, shape=(64, 16))
    xt.store(result, index=(block_id,), tile=flattened)


def test_convert_reshape_transpose_kernel():
    module = xt.convert(reshape_transpose_kernel)
    dumped = xt.dump(module)

    assert "xt.reshape" in dumped
    assert "xt.transpose" in dumped
    assert "tensor<2x32x16xf32>" in dumped
    assert "tensor<2x16x32xf32>" in dumped


@xt.kernel
def unsupported_control_flow(
    a: xt.memref("?xf32"),
    result: xt.memref("?xf32"),
):
    if 1:
        tile = xt.load(a, index=(0,), shape=(16,))
        xt.store(result, index=(0,), tile=tile)


def test_unsupported_syntax_raises():
    with pytest.raises(XTConversionError, match="unsupported statement"):
        xt.convert(unsupported_control_flow)
