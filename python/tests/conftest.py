import os
import sys


def pytest_configure() -> None:
    python_root = "/home/sjjeong94/projects/xtile/python"
    if python_root not in sys.path:
        sys.path.insert(0, python_root)
    mlir_python_root = (
        "/home/sjjeong94/projects/llvm-project/build/"
        "tools/mlir/python_packages/mlir_core"
    )
    if mlir_python_root not in sys.path:
        sys.path.insert(0, mlir_python_root)
    os.environ.setdefault("PYTHONPATH", mlir_python_root)
