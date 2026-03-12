import os
import sys


def pytest_configure() -> None:
    mlir_python_root = (
        "/home/sjjeong94/projects/llvm-project/build/"
        "tools/mlir/python_packages/mlir_core"
    )
    if mlir_python_root not in sys.path:
        sys.path.insert(0, mlir_python_root)
    os.environ.setdefault("PYTHONPATH", mlir_python_root)
