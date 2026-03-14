# CLI Xt-To-Nova Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CLI flag that applies `xt.to_nova(...)` before printing converted MLIR.

**Architecture:** Reuse the existing single-file CLI flow and add one orthogonal boolean flag. Keep the conversion pipeline linear so `convert`, `to_nova`, and `canonicalize` remain easy to reason about and test independently.

**Tech Stack:** Python stdlib `argparse`, existing `xtile.convert`, existing `xtile.to_nova`, `pytest`

---

## Chunk 1: CLI Regression Tests

### Task 1: Add failing tests for `--xt-to-nova`

**Files:**
- Modify: `python/tests/test_python_dsl.py`

- [ ] **Step 1: Write the failing tests**

Add tests that run:
- `python -m xtile python/kernels/softmax.py`
- `python -m xtile python/kernels/softmax.py --xt-to-nova`
- `python -m xtile python/kernels/softmax.py --xt-to-nova --canonicalize`

Assert the nova run contains `nova.broadcast` and `mode = 3 : i32`, while the default run still contains `xt.sub`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest python/tests/test_python_dsl.py -k xt_to_nova -v`
Expected: FAIL because the CLI does not accept `--xt-to-nova` yet.

## Chunk 2: CLI Implementation

### Task 2: Wire the new option into the CLI

**Files:**
- Modify: `python/xtile/__main__.py`

- [ ] **Step 1: Add the argparse flag**
- [ ] **Step 2: Apply `to_nova` before optional canonicalization**
- [ ] **Step 3: Keep default behavior unchanged**

- [ ] **Step 4: Run verification**

Run: `python3 -m pytest python/tests/test_python_dsl.py -k xt_to_nova -v`
Expected: PASS
