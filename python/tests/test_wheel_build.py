from __future__ import annotations

import os
import subprocess
from pathlib import Path
import sys
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_wheel.sh"


def test_build_wheel_script_produces_vendorized_wheel(tmp_path: Path):
    dist_dir = tmp_path / "dist"
    env = dict(os.environ)
    env["XTILE_DIST_DIR"] = str(dist_dir)

    subprocess.run(
        [str(SCRIPT_PATH)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    wheels = list(dist_dir.glob("*.whl"))
    assert len(wheels) == 1

    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        assert any(name.startswith("xtile/__init__.py") for name in names)
        assert any(name.startswith("mlir/_mlir_libs/_mlir") for name in names)


def test_built_wheel_imports_xtile_and_mlir(tmp_path: Path):
    dist_dir = tmp_path / "dist"
    env = dict(os.environ)
    env["XTILE_DIST_DIR"] = str(dist_dir)

    subprocess.run(
        [str(SCRIPT_PATH)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    wheel = next(dist_dir.glob("*.whl"))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--target",
            str(tmp_path / "site"),
            str(wheel),
        ],
        check=True,
    )

    script_path = tmp_path / "smoke_module.py"
    script_path.write_text(
        """
import xtile as xt
from mlir import ir

@xt.kernel
def add_kernel(a: xt.memref('?xf32'), b: xt.memref('?xf32'), result: xt.memref('?xf32')):
    tile_size = 16
    block_id = xt.bid(0)
    a_tile = xt.load(a, index=(block_id,), shape=(tile_size,))
    b_tile = xt.load(b, index=(block_id,), shape=(tile_size,))
    result_tile = a_tile + b_tile
    xt.store(result, index=(block_id,), tile=result_tile)

module = xt.convert(add_kernel)
assert isinstance(module, ir.Module)
print('ok')
""".strip()
        + "\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path / "site"), str(tmp_path)])
    subprocess.run([sys.executable, str(script_path)], env=env, check=True)
