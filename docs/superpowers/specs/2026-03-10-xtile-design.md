# xTile Design

**Date:** 2026-03-10

## Goal

Implement an out-of-tree MLIR dialect named `xt` that models tile-level NPU IR. The dialect must support parsing/printing, verification, basic canonicalization, lowering to standard MLIR dialects, and an executable optimization pipeline through a dedicated `xt-opt` tool.

## Project Layout

The project will live entirely in this repository and link against the MLIR build at `/home/sjjeong94/projects/llvm-project/build`. The structure follows the MLIR `examples/standalone` pattern:

- `include/xt`: dialect, op, pass, and TableGen definitions
- `lib/xt`: dialect implementation, op verifier/canonicalization, passes
- `xt-opt`: optimizer driver
- `test`: lit tests for parsing, verification, canonicalization, and lowering

## IR Model

`xt` distinguishes memory spaces using MLIR types:

- DRAM/global memory is represented with `memref`
- SRAM/local tile values are represented with ranked `tensor`

The initial operation set is:

- `xt.get_tile_block_id` returning three `i32` values `(x, y, z)`
- `xt.load` reading a tile from a `memref` into a `tensor`
- `xt.store` writing a tile `tensor` back to a `memref`
- `xt.add` performing elementwise tensor addition
- `xt.exp` performing elementwise tensor exponential

`xt.load` and `xt.store` require a `tile = [...]` attribute. The attribute length must match the operation rank, and that rank must match the memref rank, tensor rank, and coordinate operand count.

## Verification Rules

- `xt.get_tile_block_id` always returns exactly three `i32` values.
- `xt.load` requires a ranked memref source, `N` index-like tile coordinates, a ranked tensor result of rank `N`, and a valid `tile` attribute of length `N`.
- `xt.store` requires a ranked tensor input, ranked memref destination, `N` index-like tile coordinates, and a `tile` attribute matching the tensor shape.
- `xt.add` requires both operands and the result to have the same statically shaped ranked tensor type.
- `xt.exp` requires the operand and result to have the same statically shaped ranked tensor type.

## Canonicalization

The first implementation keeps canonicalization intentionally small:

- Fold `xt.add(%x, %zero)` and `xt.add(%zero, %x)` to `%x` when `%zero` is produced by an `arith.constant` dense splat zero tensor.
- Rely on generic canonicalization/CSE for the rest after lowering.

## Lowering Strategy

Lowering converts `xt` ops to standard MLIR dialects:

- `xt.get_tile_block_id` lowers to configurable constant `arith.constant` values via pass options. This keeps the pipeline runnable without introducing a runtime ABI.
- `xt.load` lowers to rank-generic nested `scf.for` loops that compute tile offsets and assemble the result with `tensor.insert`.
- `xt.store` lowers to rank-generic nested `scf.for` loops that extract tensor elements and write them with `memref.store`.
- `xt.add` lowers to rank-generic nested `scf.for` loops building a result tensor with `arith.addf` or `arith.addi`.
- `xt.exp` lowers to rank-generic nested `scf.for` loops using `math.exp`.

The pass pipeline target is: `xt` -> `scf` / `tensor` / `memref` / `arith` / `math`, followed by standard cleanups such as `canonicalize` and `cse`.

## Tooling

`xt-opt` will register the `xt` dialect, dependent standard dialects, and the lowering/canonicalization passes. The tool must parse the sample IR from [xtile.md](/home/sjjeong94/projects/xt/xtile.md) and lower it without any remaining `xt` operations.

## Testing

lit tests will cover:

- Parser/printer round-trip for 1D, 2D, and 3D sample IR
- Verifier failures for invalid tile rank, coordinate-count mismatch, tile shapes, and mismatched types
- Canonicalization of `xt.add` with zero tensors
- Lowering output that removes all `xt` ops in favor of standard dialects

## Notes

This repository is not currently a git repository, so the design document cannot be committed here. The implementation proceeds with local files only.
