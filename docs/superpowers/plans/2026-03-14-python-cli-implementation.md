# Python CLI Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `xtile` CLI that loads a Python file, converts every `@xt.kernel` function it defines, and prints each MLIR module to stdout.

**Architecture:** Add a small CLI module inside the `xtile` package so packaging and imports stay local to the existing Python DSL code. The CLI will load a source file with `importlib`, discover functions marked with `__xt_kernel__`, convert them in file definition order, print modules separated by blank lines, and return a non-zero exit code only when no kernels are found or the file cannot be loaded.

**Tech Stack:** Python stdlib (`argparse`, `importlib`, `pathlib`, `inspect`), `pytest`, existing `xtile.convert` / `xtile.dump`

---

## Chunk 1: CLI Tests

### Task 1: Add failing CLI tests

**Files:**
- Modify: `python/tests/test_python_dsl.py`
- Test: `python/tests/test_python_dsl.py`

- [ ] **Step 1: Write the failing test**

Add tests that invoke `python -m xtile <file.py>` through `subprocess.run(...)` and assert:
- one kernel file prints one `func.func @...`
- multi-kernel file prints multiple `module {` blocks in source order
- a file with no `@xt.kernel` exits non-zero with a useful stderr message

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest python/tests/test_python_dsl.py -k 'cli' -q`
Expected: FAIL because `python -m xtile` does not exist yet

### Task 2: Implement minimal CLI

**Files:**
- Create: `python/xtile/__main__.py`
- Modify: `python/xtile/__init__.py`

- [ ] **Step 3: Write minimal implementation**

Implement a CLI entrypoint that:
- parses a single file path argument
- loads the file as a module
- finds every callable with `__xt_kernel__ == True`
- converts each with `xtile.convert`
- prints `xtile.dump(module)` results separated by one blank line
- exits with status `1` and stderr text if no kernels are found

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest python/tests/test_python_dsl.py -k 'cli' -q`
Expected: PASS

## Chunk 2: Packaging and Regression Coverage

### Task 3: Expose installed `xtile` command

**Files:**
- Modify: `pyproject.toml`
- Modify: `python/tests/test_wheel_build.py`

- [ ] **Step 5: Write the failing packaging test**

Add a wheel smoke test that installs the built wheel and runs the `xtile` console script against a small kernel file.

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest python/tests/test_wheel_build.py -k 'console' -q`
Expected: FAIL because no console script is registered

- [ ] **Step 7: Write minimal packaging implementation**

Add a `project.scripts` entry mapping `xtile` to the CLI module.

- [ ] **Step 8: Run targeted tests to verify they pass**

Run: `pytest python/tests/test_python_dsl.py -k 'cli' -q`
Expected: PASS

Run: `pytest python/tests/test_wheel_build.py -k 'console' -q`
Expected: PASS

### Task 4: Full verification

**Files:**
- Modify: `python/tests/conftest.py` only if CLI subprocesses need explicit path setup

- [ ] **Step 9: Run DSL regression suite**

Run: `pytest python/tests/test_python_dsl.py -q`
Expected: PASS

- [ ] **Step 10: Run wheel build regression suite**

Run: `pytest python/tests/test_wheel_build.py -q`
Expected: PASS
