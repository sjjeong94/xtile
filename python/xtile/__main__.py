from __future__ import annotations

import argparse
from collections.abc import Iterable
import importlib.util
import inspect
from pathlib import Path
import sys

from mlir import passmanager

from . import convert, dump, to_nova


def _load_module(source_path: Path) -> object:
    spec = importlib.util.spec_from_file_location("_xtile_cli_module", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load Python source file: {source_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _iter_kernels(module: object) -> Iterable[object]:
    functions = inspect.getmembers(module, inspect.isfunction)
    for _, fn in sorted(functions, key=lambda item: item[1].__code__.co_firstlineno):
        if getattr(fn, "__xt_kernel__", False):
            yield fn


def _canonicalize(module: object) -> None:
    with module.context:
        pm = passmanager.PassManager.parse("builtin.module(canonicalize)")
        pm.run(module.operation)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xtile")
    parser.add_argument("source", help="Python file containing @xt.kernel definitions")
    parser.add_argument(
        "--canonicalize",
        action="store_true",
        help="run the MLIR canonicalize pass before printing",
    )
    parser.add_argument(
        "--xt-to-nova",
        action="store_true",
        help="convert supported xt binary ops to nova ops before printing",
    )
    args = parser.parse_args(argv)

    source_path = Path(args.source).resolve()
    try:
        module = _load_module(source_path)
        kernels = list(_iter_kernels(module))
        if not kernels:
            print("no @xt.kernel functions found", file=sys.stderr)
            return 1

        modules = []
        for fn in kernels:
            converted = convert(fn)
            if args.xt_to_nova:
                converted = to_nova(converted)
            if args.canonicalize:
                _canonicalize(converted)
            modules.append(dump(converted))
        print("\n\n".join(modules))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
