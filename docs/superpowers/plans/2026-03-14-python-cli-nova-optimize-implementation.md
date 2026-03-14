# Python CLI Nova Optimize Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--nova-optimize` flag to the Python CLI and run the existing Nova optimization pass from Python.

**Architecture:** Extend the CLI argument parser, add a small helper that runs `builtin.module(func.func(nova-optimize))` through the MLIR Python `PassManager`, and place the step after `xt.to_nova()` and before canonicalization. Verify with CLI regression tests that the folded output appears.

**Tech Stack:** Python argparse, MLIR Python PassManager, pytest

---

## Chunk 1: Tests First

### Task 1: Add failing CLI coverage

**Files:**
- Modify: `python/tests/test_python_dsl.py`

- [ ] **Step 1: Write the failing test**
  Add a CLI test that invokes `xtile` with `--xt-to-nova --nova-optimize` and expects folded Nova output.

- [ ] **Step 2: Run test to verify it fails**
  Run: `python3 -m pytest python/tests/test_python_dsl.py -k nova_optimize -v`
  Expected: FAIL because the CLI flag does not exist yet.

## Chunk 2: CLI Implementation

### Task 2: Add the CLI flag and pass runner

**Files:**
- Modify: `python/xtile/__main__.py`

- [ ] **Step 1: Add the parser flag**
  Add `--nova-optimize` to the CLI.

- [ ] **Step 2: Add a helper to run the MLIR pass**
  Use `PassManager.parse("builtin.module(func.func(nova-optimize))")`.

- [ ] **Step 3: Integrate the new pipeline step**
  Run the pass after `to_nova()` and before canonicalization.

## Chunk 3: Verification

### Task 3: Run focused verification

**Files:**
- Test: `python/tests/test_python_dsl.py`

- [ ] **Step 1: Run the new CLI regression**
  Run: `python3 -m pytest python/tests/test_python_dsl.py -k nova_optimize -v`
  Expected: PASS

### Task 4: Run full verification

**Files:**
- Test: `./scripts/build.sh`

- [ ] **Step 1: Run the full build and test script**
  Run: `./scripts/build.sh`
  Expected: build succeeds, lit passes, pytest passes.
