# CLI Canonicalize Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--canonicalize` to the `xtile` CLI so it prints canonicalized MLIR after conversion.

**Architecture:** Keep canonicalization as a CLI-only concern. The CLI will optionally run MLIR's `canonicalize` pass over each converted module before dumping it, avoiding any changes to the core `xtile.convert` API.

**Tech Stack:** Python stdlib, MLIR Python bindings (`mlir.passmanager`), existing `xtile` CLI tests

---

## Chunk 1: Tests First

### Task 1: Add failing CLI canonicalization tests

**Files:**
- Modify: `python/tests/test_python_dsl.py`
- Test: `python/tests/test_python_dsl.py`

- [ ] **Step 1: Write the failing test**

Add a CLI subprocess test that runs the same temporary kernel file twice:
- once with `python -m xtile <file>`
- once with `python -m xtile <file> --canonicalize`

Assert the canonicalized output succeeds and differs in at least one expected way from the raw output.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest python/tests/test_python_dsl.py -k 'canonicalize' -q`
Expected: FAIL because the CLI does not recognize `--canonicalize`

## Chunk 2: Minimal Implementation

### Task 2: Implement CLI canonicalization

**Files:**
- Modify: `python/xtile/__main__.py`

- [ ] **Step 3: Write minimal implementation**

Add:
- a `--canonicalize` flag to the CLI parser
- a helper that runs `builtin.module(canonicalize)` through `mlir.passmanager.PassManager`
- conditional canonicalization before dumping each module

- [ ] **Step 4: Run targeted tests to verify they pass**

Run: `pytest python/tests/test_python_dsl.py -k 'canonicalize' -q`
Expected: PASS

## Chunk 3: Full Verification

### Task 3: Run regression checks

**Files:**
- No additional file changes expected

- [ ] **Step 5: Run DSL regression suite**

Run: `pytest python/tests/test_python_dsl.py -q`
Expected: PASS

- [ ] **Step 6: Run wheel regression suite**

Run: `pytest python/tests/test_wheel_build.py -q`
Expected: PASS
