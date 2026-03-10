# xTile Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an out-of-tree MLIR `xt` dialect project with parser/printer support, verification, canonicalization, lowering passes, a dedicated `xt-opt` driver, and lit tests covering the sample IR.

**Architecture:** The repository mirrors MLIR's standalone example and defines `xt` operations through ODS/TableGen plus targeted C++ verification and rewrite logic. Lowering converts tile operations into nested `scf` loops and core MLIR dialects so the resulting IR is executable by standard downstream pipelines.

**Tech Stack:** CMake, TableGen, C++17, MLIR/LLVM, lit/FileCheck

---

## Chunk 1: Project Scaffold

### Task 1: Create the out-of-tree project skeleton

**Files:**
- Create: `CMakeLists.txt`
- Create: `include/CMakeLists.txt`
- Create: `include/xt/CMakeLists.txt`
- Create: `lib/CMakeLists.txt`
- Create: `lib/xt/CMakeLists.txt`
- Create: `xt-opt/CMakeLists.txt`
- Create: `test/CMakeLists.txt`
- Create: `test/lit.cfg.py`
- Create: `test/lit.site.cfg.py.in`

- [ ] **Step 1: Write the failing build-oriented smoke test**

Create a parse test in `test/xt/parse.mlir` that references `xt` ops and `xt-opt`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake -G Ninja -S . -B build -DMLIR_DIR=/home/sjjeong94/projects/llvm-project/build/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=/home/sjjeong94/projects/llvm-project/build/bin/llvm-lit && cmake --build build --target check-xt`
Expected: FAIL because the project files and target do not exist yet.

- [ ] **Step 3: Write minimal scaffold**

Copy the `standalone` project pattern, rename targets to `xt`, and wire lit configuration.

- [ ] **Step 4: Run build to verify scaffold works**

Run: `cmake -G Ninja -S . -B build -DMLIR_DIR=/home/sjjeong94/projects/llvm-project/build/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=/home/sjjeong94/projects/llvm-project/build/bin/llvm-lit`
Expected: configure succeeds.

### Task 2: Define the dialect and operation TableGen files

**Files:**
- Create: `include/xt/XTDialect.h`
- Create: `include/xt/XTDialect.td`
- Create: `include/xt/XTOps.h`
- Create: `include/xt/XTOps.td`
- Create: `include/xt/XTPasses.h`
- Create: `include/xt/XTPasses.td`

- [ ] **Step 1: Write the failing parse test**

Extend `test/xt/parse.mlir` to require successful parse/print of the example ops.

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `build/bin/xt-opt test/xt/parse.mlir`
Expected: FAIL because `xt` dialect and ops are undefined.

- [ ] **Step 3: Implement the dialect declarations**

Define the dialect and ODS operation declarations for `xt.get_tile_block_id`, `xt.load`, `xt.store`, `xt.add`, and `xt.exp`.

- [ ] **Step 4: Rebuild and verify parse progresses**

Run: `cmake --build build --target xt-opt`
Expected: build succeeds or advances to missing C++ implementations.

## Chunk 2: Dialect Implementation

### Task 3: Implement dialect/op registration and verification

**Files:**
- Create: `lib/xt/XTDialect.cpp`
- Create: `lib/xt/XTOps.cpp`

- [ ] **Step 1: Write the failing verifier tests**

Create `test/xt/invalid.mlir` with type and tile-shape mismatches.

- [ ] **Step 2: Run verifier tests to verify they fail correctly**

Run: `build/bin/xt-opt test/xt/invalid.mlir`
Expected: FAIL with verifier diagnostics tied to the invalid ops.

- [ ] **Step 3: Implement verification and canonicalization hooks**

Add custom verifiers for tile attributes, tensor shapes, and element types. Add canonicalization for zero-add folding.

- [ ] **Step 4: Run parser and verifier tests**

Run: `cmake --build build --target check-xt`
Expected: parse and verifier tests pass.

## Chunk 3: Lowering

### Task 4: Implement the xt lowering pass infrastructure

**Files:**
- Create: `lib/xt/XTLowerToLoops.cpp`
- Modify: `include/xt/XTPasses.h`
- Modify: `include/xt/XTPasses.td`
- Modify: `lib/xt/CMakeLists.txt`

- [ ] **Step 1: Write the failing lowering test**

Create `test/xt/lower.mlir` with FileCheck assertions that no `xt.` ops remain after the pass.

- [ ] **Step 2: Run lowering test to verify it fails**

Run: `build/bin/xt-opt --xt-lower-to-loops test/xt/lower.mlir | /home/sjjeong94/projects/llvm-project/build/bin/FileCheck test/xt/lower.mlir`
Expected: FAIL because the pass does not exist.

- [ ] **Step 3: Implement minimal lowering**

Lower all `xt` ops to `scf`, `tensor`, `memref`, `arith`, and `math`. Support pass options for constant tile block ids.

- [ ] **Step 4: Run lowering tests**

Run: `cmake --build build --target check-xt`
Expected: lowering tests pass.

## Chunk 4: Driver and Verification

### Task 5: Implement the `xt-opt` driver and full verification

**Files:**
- Create: `xt-opt/xt-opt.cpp`
- Modify: `xt-opt/CMakeLists.txt`
- Modify: `test/CMakeLists.txt`
- Create: `test/xt/parse.mlir`
- Create: `test/xt/invalid.mlir`
- Create: `test/xt/canonicalize.mlir`
- Create: `test/xt/lower.mlir`

- [ ] **Step 1: Write the failing canonicalization test**

Add `test/xt/canonicalize.mlir` expecting zero-add folding under `--canonicalize`.

- [ ] **Step 2: Run canonicalization test to verify it fails**

Run: `build/bin/xt-opt --canonicalize test/xt/canonicalize.mlir | /home/sjjeong94/projects/llvm-project/build/bin/FileCheck test/xt/canonicalize.mlir`
Expected: FAIL because the pattern is not registered or the op is not canonicalized.

- [ ] **Step 3: Register passes/dialects and finish test coverage**

Wire the tool, test suite, and pass registration.

- [ ] **Step 4: Run full verification**

Run: `cmake --build build --target check-xt`
Expected: all tests pass.

- [ ] **Step 5: Record completion status**

If the repository becomes a git repository later, commit with:

```bash
git add .
git commit -m "feat: implement xt MLIR dialect project"
```

## Chunk 5: Rank-Generic Tile Support

### Task 6: Generalize load/store syntax and verification

**Files:**
- Modify: `include/xt/XTOps.td`
- Modify: `lib/xt/XTOps.cpp`
- Modify: `test/xt/parse.mlir`
- Modify: `test/xt/invalid.mlir`

- [ ] **Step 1: Write the failing tests**

Add 1D and 3D parse coverage plus invalid rank/coordinate mismatch cases.

- [ ] **Step 2: Run tests to verify they fail**

Run: `build/bin/xt-opt test/xt/parse.mlir`
Expected: FAIL on 1D/3D examples because the current implementation is rank-2 only.

- [ ] **Step 3: Implement rank-generic parser and verifier logic**

Allow variadic coordinates for `xt.load` and `xt.store`, generalize `tile=[...]`, and validate that tile rank, coordinate count, tensor rank, and memref rank all match.

- [ ] **Step 4: Run focused verification**

Run: `build/bin/xt-opt test/xt/parse.mlir`
Expected: PASS for 1D, 2D, and 3D examples.

### Task 7: Generalize lowering and elementwise ops

**Files:**
- Modify: `lib/xt/XTLowerToLoops.cpp`
- Modify: `lib/xt/XTOps.cpp`
- Modify: `test/xt/lower.mlir`
- Modify: `test/xt/canonicalize.mlir`

- [ ] **Step 1: Write the failing tests**

Add 1D and 3D lowering coverage and a non-2D canonicalization case.

- [ ] **Step 2: Run tests to verify they fail**

Run: `build/bin/xt-opt --xt-lower-to-loops test/xt/lower.mlir`
Expected: FAIL or produce incorrect IR because the lowering is rank-2 specific.

- [ ] **Step 3: Implement rank-generic lowering**

Build loop nests from the tensor rank and lower all elementwise and memory ops through the same rank-generic path.

- [ ] **Step 4: Run full verification**

Run: `cmake --build build --target check-xt`
Expected: all tests pass for 1D, 2D, and 3D examples.

## Chunk 6: Extended Compute Ops

### Task 8: Add new elementwise and contraction op declarations

**Files:**
- Modify: `include/xt/XTOps.td`
- Modify: `lib/xt/XTOps.cpp`
- Modify: `test/xt/parse.mlir`
- Modify: `test/xt/invalid.mlir`

- [ ] **Step 1: Write the failing tests**

Add parse and invalid coverage for `sub`, `mul`, `matmul`, `mma`, `cos`, `sin`, `reciprocal`, `rsqrt`, `sigmoid`, `tanh`, and `silu`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `build/bin/xt-opt test/xt/parse.mlir`
Expected: FAIL because the new ops do not exist yet.

- [ ] **Step 3: Implement op declarations and verifiers**

Define the new ops and enforce the requested `mma` type restrictions plus `matmul`/`mma` shape rules.

- [ ] **Step 4: Run focused verification**

Run: `build/bin/xt-opt test/xt/parse.mlir && not build/bin/xt-opt test/xt/invalid.mlir`
Expected: new valid parse cases succeed and invalid cases fail with the intended diagnostics.

### Task 9: Lower new ops and verify the full pipeline

**Files:**
- Modify: `lib/xt/XTLowerToLoops.cpp`
- Modify: `test/xt/lower.mlir`
- Modify: `test/xt/canonicalize.mlir`

- [ ] **Step 1: Write the failing lowering tests**

Add representative lowering coverage for unary ops, binary ops, `matmul`, and `mma`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `build/bin/xt-opt --xt-lower-to-loops test/xt/lower.mlir`
Expected: FAIL or produce incorrect IR because the new lowering patterns do not exist yet.

- [ ] **Step 3: Implement minimal lowering**

Lower the new elementwise ops through rank-generic loops and lower `matmul`/`mma` through rank-2 reduction loops with the requested `mma` type conversions.

- [ ] **Step 4: Run full verification**

Run: `cmake --build build --target check-xt`
Expected: all tests pass with the extended operation set.
