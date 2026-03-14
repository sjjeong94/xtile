# Nova Optimize Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--nova-optimize` pass that folds `nova.scalar` into `nova.broadcast` and `nova.elementwise`.

**Architecture:** Create a Nova-specific function pass and register it in `xt-opt`. The pass pattern-matches Nova binary ops, inspects `lhs` and `rhs` defining ops for `nova.scalar`, rewrites supported add/mul scalar modes into the corresponding scale/bias attributes, and leaves unsupported cases untouched.

**Tech Stack:** MLIR TableGen/C++, lit/FileCheck

---

## Chunk 1: Tests First

### Task 1: Add failing lit coverage

**Files:**
- Create: `test/nova/optimize.mlir`

- [ ] **Step 1: Write the failing test**
  Add cases for rhs fold, lhs fold, both-side fold, and non-folded sub mode.

- [ ] **Step 2: Run test to verify it fails**
  Run: `./build/bin/xt-opt --nova-optimize test/nova/optimize.mlir | /home/sjjeong94/projects/llvm-project/build/bin/FileCheck test/nova/optimize.mlir`
  Expected: FAIL because the pass does not exist yet.

## Chunk 2: Pass Plumbing

### Task 2: Add Nova pass declarations

**Files:**
- Create: `include/nova/NovaPasses.td`
- Create: `include/nova/NovaPasses.h`
- Modify: `include/nova/CMakeLists.txt`

- [ ] **Step 1: Define the pass TableGen entry**
  Add a `NovaOptimize` pass on `::mlir::func::FuncOp`.

- [ ] **Step 2: Expose pass creation and registration**
  Add a C++ header mirroring the existing XT pass pattern.

### Task 3: Wire build and driver registration

**Files:**
- Modify: `lib/nova/CMakeLists.txt`
- Modify: `xt-opt/xt-opt.cpp`

- [ ] **Step 1: Build the new pass source**
  Add the implementation file and generated pass dependency.

- [ ] **Step 2: Register Nova passes in the driver**
  Include the new header and call `mlir::nova::registerPasses()`.

## Chunk 3: Pass Implementation

### Task 4: Implement folding logic

**Files:**
- Create: `lib/nova/NovaOptimize.cpp`

- [ ] **Step 1: Implement a failing skeleton if needed**
  Add the pass class and pattern container.

- [ ] **Step 2: Implement scalar-fold helper**
  Convert supported `nova.scalar` add/mul modes into updated scale/bias pairs.

- [ ] **Step 3: Implement the rewrite**
  Fold supported `lhs`/`rhs` scalar producers into `nova.broadcast` and `nova.elementwise`.

## Chunk 4: Verification

### Task 5: Run focused verification

**Files:**
- Test: `test/nova/optimize.mlir`

- [ ] **Step 1: Run the new lit test**
  Run: `./build/bin/xt-opt --nova-optimize test/nova/optimize.mlir | /home/sjjeong94/projects/llvm-project/build/bin/FileCheck test/nova/optimize.mlir`
  Expected: PASS

### Task 6: Run full verification

**Files:**
- Test: `./scripts/build.sh`

- [ ] **Step 1: Run the full build and test script**
  Run: `./scripts/build.sh`
  Expected: build succeeds, lit passes, pytest passes.
