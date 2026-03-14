import xtile as xt

from mlir import ir
import os
from pathlib import Path
import pytest
import subprocess
import sys
from kernels.softmax import softmax_kernel

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
def cos_kernel(
    a: xt.memref("?xf32"),
    result: xt.memref("?xf32"),
):
    tile_size = 16
    block_id = xt.bid(0)
    a_tile = xt.load(a, index=(block_id,), shape=(tile_size,))
    result_tile = xt.cos(a_tile)
    xt.store(result, index=(block_id,), tile=result_tile)


@xt.kernel
def sin_kernel(
    a: xt.memref("?xf32"),
    result: xt.memref("?xf32"),
):
    tile_size = 16
    block_id = xt.bid(0)
    a_tile = xt.load(a, index=(block_id,), shape=(tile_size,))
    result_tile = xt.sin(a_tile)
    xt.store(result, index=(block_id,), tile=result_tile)


@xt.kernel
def reciprocal_kernel(
    a: xt.memref("?xf32"),
    result: xt.memref("?xf32"),
):
    tile_size = 16
    block_id = xt.bid(0)
    a_tile = xt.load(a, index=(block_id,), shape=(tile_size,))
    result_tile = xt.reciprocal(a_tile)
    xt.store(result, index=(block_id,), tile=result_tile)


@xt.kernel
def rsqrt_kernel(
    a: xt.memref("?xf32"),
    result: xt.memref("?xf32"),
):
    tile_size = 16
    block_id = xt.bid(0)
    a_tile = xt.load(a, index=(block_id,), shape=(tile_size,))
    result_tile = xt.rsqrt(a_tile)
    xt.store(result, index=(block_id,), tile=result_tile)


@xt.kernel
def sigmoid_kernel(
    a: xt.memref("?xf32"),
    result: xt.memref("?xf32"),
):
    tile_size = 16
    block_id = xt.bid(0)
    a_tile = xt.load(a, index=(block_id,), shape=(tile_size,))
    result_tile = xt.sigmoid(a_tile)
    xt.store(result, index=(block_id,), tile=result_tile)


@xt.kernel
def tanh_kernel(
    a: xt.memref("?xf32"),
    result: xt.memref("?xf32"),
):
    tile_size = 16
    block_id = xt.bid(0)
    a_tile = xt.load(a, index=(block_id,), shape=(tile_size,))
    result_tile = xt.tanh(a_tile)
    xt.store(result, index=(block_id,), tile=result_tile)


@xt.kernel
def silu_kernel(
    a: xt.memref("?xf32"),
    result: xt.memref("?xf32"),
):
    tile_size = 16
    block_id = xt.bid(0)
    a_tile = xt.load(a, index=(block_id,), shape=(tile_size,))
    result_tile = xt.silu(a_tile)
    xt.store(result, index=(block_id,), tile=result_tile)


@pytest.mark.parametrize(
    ("op_name", "kernel"),
    [
        ("exp", exp_kernel),
        ("cos", cos_kernel),
        ("sin", sin_kernel),
        ("reciprocal", reciprocal_kernel),
        ("rsqrt", rsqrt_kernel),
        ("sigmoid", sigmoid_kernel),
        ("tanh", tanh_kernel),
        ("silu", silu_kernel),
    ],
)
def test_convert_supported_unary_kernels(op_name: str, kernel: object):
    module = xt.convert(kernel)
    dumped = xt.dump(module)

    assert f"xt.{op_name}" in dumped
    assert "tensor<16xf32>" in dumped


@xt.kernel
def sum_kernel(
    a: xt.memref("?xf32"),
    result: xt.memref("?xf32"),
):
    block_id = xt.bid(0)
    a_tile = xt.load(a, index=(block_id,), shape=(16, 16))
    result_tile = xt.sum(a_tile)
    xt.store(result, index=(block_id,), tile=result_tile)


@xt.kernel
def max_kernel(
    a: xt.memref("?xf32"),
    result: xt.memref("?xf32"),
):
    block_id = xt.bid(0)
    a_tile = xt.load(a, index=(block_id,), shape=(16, 16))
    result_tile = xt.max(a_tile)
    xt.store(result, index=(block_id,), tile=result_tile)


@pytest.mark.parametrize(
    ("op_name", "kernel"),
    [
        ("reduce_sum", sum_kernel),
        ("reduce_max", max_kernel),
    ],
)
def test_convert_supported_reduce_kernels(op_name: str, kernel: object):
    module = xt.convert(kernel)
    dumped = xt.dump(module)

    assert f"xt.{op_name}" in dumped
    assert "tensor<16x1xf32>" in dumped


@xt.kernel
def broadcast_sub_kernel(
    a: xt.memref("?xf32"),
    result: xt.memref("?xf32"),
):
    block_id = xt.bid(0)
    a_tile = xt.load(a, index=(block_id,), shape=(16, 16))
    row_max = xt.max(a_tile)
    shifted = a_tile - row_max
    xt.store(result, index=(block_id,), tile=shifted)


def test_convert_broadcast_binary_kernel():
    module = xt.convert(broadcast_sub_kernel)
    dumped = xt.dump(module)

    assert "xt.reduce_max" in dumped
    assert "xt.sub" in dumped
    assert "tensor<16x16xf32>" in dumped
    assert "tensor<16x1xf32>" in dumped


@xt.kernel
def elementwise_mul_kernel(
    a: xt.memref("?xf32"),
    b: xt.memref("?xf32"),
    result: xt.memref("?xf32"),
):
    block_id = xt.bid(0)
    lhs = xt.load(a, index=(block_id,), shape=(16, 16))
    rhs = xt.load(b, index=(block_id,), shape=(16, 16))
    prod = lhs * rhs
    xt.store(result, index=(block_id,), tile=prod)


def test_convert_then_to_nova_broadcast_binary_kernel():
    module = xt.convert(broadcast_sub_kernel)
    module = xt.to_nova(module)
    dumped = xt.dump(module)

    assert "nova.reduce" in dumped
    assert "mode = 1 : i32" in dumped
    assert "nova.broadcast" in dumped
    assert "mode = 3 : i32" in dumped
    assert "xt.sub" not in dumped


def test_convert_then_to_nova_elementwise_kernel():
    module = xt.convert(elementwise_mul_kernel)
    module = xt.to_nova(module)
    dumped = xt.dump(module)

    assert "nova.elementwise" in dumped
    assert "mode = 2 : i32" in dumped
    assert "xt.mul" not in dumped


def test_convert_then_to_nova_skips_scalar_like_broadcast():
    from kernels.layernorm import layernorm_kernel

    module = xt.convert(layernorm_kernel)
    module = xt.to_nova(module)
    dumped = xt.dump(module)

    assert "nova.reduce" in dumped
    assert dumped.count("nova.scalar") == 2
    assert "mode = 2 : i32" in dumped
    assert "rhs = 6.250000e-02 : f32" in dumped
    assert '"xt.mul"(%2, %cst)' not in dumped
    assert '"xt.mul"(%6, %cst_0)' not in dumped


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
def matmul_kernel(
    a: xt.memref("?x128xf32"),
    b: xt.memref("128x?xf32"),
    result: xt.memref("?x?xf32"),
):
    tm, tn = 64, 64
    bid_x = xt.bid(0)
    bid_y = xt.bid(1)

    a_tile = xt.load(a, index=(bid_x, 0), shape=(tm, 128))
    b_tile = xt.load(b, index=(0, bid_y), shape=(128, tn))
    result_tile = a_tile @ b_tile
    xt.store(result, index=(bid_x, bid_y), tile=result_tile)


def test_convert_matmul_kernel():
    module = xt.convert(matmul_kernel)
    dumped = xt.dump(module)

    assert "func.func @matmul_kernel" in dumped
    assert dumped.count("xt.load") == 2
    assert "xt.matmul" in dumped
    assert "xt.store" in dumped
    assert "tensor<64x64xf32>" in dumped


def test_convert_matmul_kernel_example():
    from kernels.matmul import matmul_kernel as example_matmul_kernel

    module = xt.convert(example_matmul_kernel)
    dumped = xt.dump(module)

    assert "func.func @matmul_kernel" in dumped
    assert dumped.count("xt.load") == 2
    assert "xt.matmul" in dumped
    assert "xt.store" in dumped
    assert "tensor<64x64xf32>" in dumped


def test_convert_softmax_kernel():
    module = xt.convert(softmax_kernel)
    dumped = xt.dump(module)

    assert "func.func @softmax_kernel" in dumped
    assert "xt.reduce_max" in dumped
    assert "xt.sub" in dumped
    assert "xt.exp" in dumped
    assert "xt.reduce_sum" in dumped
    assert "xt.reciprocal" in dumped
    assert "xt.mul" in dumped
    assert "tensor<16x16xf32>" in dumped
    assert "tensor<16x1xf32>" in dumped


@xt.kernel
def constant_tensor_scale_kernel(
    a: xt.memref("?xf32"),
    result: xt.memref("?xf32"),
):
    block_id = xt.bid(0)
    a_tile = xt.load(a, index=(block_id,), shape=(16, 16))
    scale = xt.full(shape=(16, 16), value=0.5)
    scaled = a_tile * scale
    xt.store(result, index=(block_id,), tile=scaled)


def test_convert_constant_tensor_kernel():
    module = xt.convert(constant_tensor_scale_kernel)
    dumped = xt.dump(module)

    assert "arith.constant" in dumped
    assert "dense<5.000000e-01>" in dumped
    assert "xt.mul" in dumped
    assert "tensor<16x16xf32>" in dumped


def test_convert_layernorm_kernel():
    from kernels.layernorm import layernorm_kernel

    module = xt.convert(layernorm_kernel)
    dumped = xt.dump(module)

    assert "func.func @layernorm_kernel" in dumped
    assert dumped.count("xt.reduce_sum") == 2
    assert dumped.count("xt.mul") >= 3
    assert "xt.sub" in dumped
    assert "xt.rsqrt" in dumped
    assert "arith.constant" in dumped
    assert "6.250000e-02" in dumped
    assert "tensor<1x1xf32>" in dumped
    assert "tensor<16x16xf32>" in dumped
    assert "tensor<16x1xf32>" in dumped


def _run_xtile_cli(
    source_path: Path, *extra_args: str
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    python_root = "/home/sjjeong94/projects/xtile/python"
    mlir_python_root = (
        "/home/sjjeong94/projects/llvm-project/build/"
        "tools/mlir/python_packages/mlir_core"
    )
    env["PYTHONPATH"] = os.pathsep.join([python_root, mlir_python_root])
    return subprocess.run(
        [sys.executable, "-m", "xtile", str(source_path), *extra_args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_cli_prints_single_kernel_mlir(tmp_path: Path):
    source_path = tmp_path / "single_kernel.py"
    source_path.write_text(
        """
import xtile as xt

@xt.kernel
def only_kernel(a: xt.memref('?xf32'), result: xt.memref('?xf32')):
    block_id = xt.bid(0)
    tile = xt.load(a, index=(block_id,), shape=(16,))
    xt.store(result, index=(block_id,), tile=tile)
""".strip()
        + "\n"
    )

    completed = _run_xtile_cli(source_path)

    assert completed.returncode == 0
    assert "func.func @only_kernel" in completed.stdout
    assert completed.stderr == ""


def test_cli_prints_all_kernels_in_source_order(tmp_path: Path):
    source_path = tmp_path / "multiple_kernels.py"
    source_path.write_text(
        """
import xtile as xt

@xt.kernel
def first_kernel(a: xt.memref('?xf32'), result: xt.memref('?xf32')):
    block_id = xt.bid(0)
    tile = xt.load(a, index=(block_id,), shape=(16,))
    xt.store(result, index=(block_id,), tile=tile)

@xt.kernel
def second_kernel(a: xt.memref('?xf32'), result: xt.memref('?xf32')):
    block_id = xt.bid(0)
    tile = xt.load(a, index=(block_id,), shape=(16,))
    shifted = xt.exp(tile)
    xt.store(result, index=(block_id,), tile=shifted)
""".strip()
        + "\n"
    )

    completed = _run_xtile_cli(source_path)

    assert completed.returncode == 0
    assert completed.stdout.count("module {") == 2
    assert completed.stdout.index("func.func @first_kernel") < completed.stdout.index(
        "func.func @second_kernel"
    )


def test_cli_errors_when_file_has_no_kernels(tmp_path: Path):
    source_path = tmp_path / "no_kernel.py"
    source_path.write_text(
        """
def helper():
    return 0
""".strip()
        + "\n"
    )

    completed = _run_xtile_cli(source_path)

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "no @xt.kernel functions found" in completed.stderr


def test_cli_canonicalize_prints_canonicalized_mlir():
    source_path = Path("/home/sjjeong94/projects/xtile/python/kernels/matmul.py")

    raw = _run_xtile_cli(source_path)
    canonicalized = _run_xtile_cli(source_path, "--canonicalize")

    assert raw.returncode == 0
    assert canonicalized.returncode == 0
    assert "%c64_i32 = arith.constant 64 : i32" in raw.stdout
    assert "%c64_i32 = arith.constant 64 : i32" not in canonicalized.stdout
    assert raw.stdout.count("arith.constant 0 : i32") == 2
    assert canonicalized.stdout.count("arith.constant 0 : i32") == 1


def test_cli_xt_to_nova_prints_nova_mlir():
    source_path = Path("/home/sjjeong94/projects/xtile/python/kernels/softmax.py")

    raw = _run_xtile_cli(source_path)
    nova = _run_xtile_cli(source_path, "--xt-to-nova")

    assert raw.returncode == 0
    assert nova.returncode == 0
    assert "xt.sub" in raw.stdout
    assert "nova.reduce" in nova.stdout
    assert "mode = 1 : i32" in nova.stdout
    assert "mode = 0 : i32" in nova.stdout
    assert "nova.broadcast" in nova.stdout
    assert "mode = 3 : i32" in nova.stdout
    assert "xt.sub" not in nova.stdout


def test_cli_xt_to_nova_and_canonicalize_work_together():
    source_path = Path("/home/sjjeong94/projects/xtile/python/kernels/softmax.py")

    completed = _run_xtile_cli(source_path, "--xt-to-nova", "--canonicalize")

    assert completed.returncode == 0
    assert "nova.reduce" in completed.stdout
    assert "nova.broadcast" in completed.stdout


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
