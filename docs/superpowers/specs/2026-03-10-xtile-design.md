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
- `xt.sub` performing elementwise tensor subtraction
- `xt.mul` performing elementwise tensor multiplication
- `xt.exp` performing elementwise tensor exponential
- `xt.cos`, `xt.sin`, `xt.reciprocal`, `xt.rsqrt`, `xt.sigmoid`, `xt.tanh`, and `xt.silu` performing elementwise tensor math
- `xt.matmul` performing rank-2 matrix multiplication
- `xt.mma` performing rank-2 matrix multiply-accumulate with `i8` inputs and `f32`/`bf16` accumulator+result tensors

`xt.load` and `xt.store` require a `tile = [...]` attribute. The attribute length must match the operation rank, and that rank must match the memref rank, tensor rank, and coordinate operand count.

## Verification Rules

- `xt.get_tile_block_id` always returns exactly three `i32` values.
- `xt.load` requires a ranked memref source, `N` index-like tile coordinates, a ranked tensor result of rank `N`, and a valid `tile` attribute of length `N`.
- `xt.store` requires a ranked tensor input, ranked memref destination, `N` index-like tile coordinates, and a `tile` attribute matching the tensor shape.
- `xt.add` requires both operands and the result to have the same statically shaped ranked tensor type.
- `xt.sub` and `xt.mul` require both operands and the result to have the same statically shaped ranked tensor type.
- `xt.exp` requires the operand and result to have the same statically shaped ranked tensor type.
- Unary elementwise math ops require the operand and result to have the same statically shaped ranked tensor type.
- `xt.matmul` requires rank-2 tensors with shapes `MxK`, `KxN`, and `MxN`.
- `xt.mma` requires rank-2 tensors with shapes `MxK`, `KxN`, `MxN`, and `MxN`; input element types must be `i8`, accumulator/result element types must be `f32` or `bf16`, and accumulator/result types must match.

## Canonicalization

The first implementation keeps canonicalization intentionally small:

- Fold `xt.add(%x, %zero)` and `xt.add(%zero, %x)` to `%x` when `%zero` is produced by an `arith.constant` dense splat zero tensor.
- Fold the same zero pattern for `xt.sub(%x, %zero)` to `%x`.
- Rely on generic canonicalization/CSE for the rest after lowering.

## Lowering Strategy

Lowering converts `xt` ops to standard MLIR dialects:

- `xt.get_tile_block_id` lowers to configurable constant `arith.constant` values via pass options. This keeps the pipeline runnable without introducing a runtime ABI.
- `xt.load` lowers to rank-generic nested `scf.for` loops that compute tile offsets and assemble the result with `tensor.insert`.
- `xt.store` lowers to rank-generic nested `scf.for` loops that extract tensor elements and write them with `memref.store`.
- `xt.add` lowers to rank-generic nested `scf.for` loops building a result tensor with `arith.addf` or `arith.addi`.
- `xt.sub` and `xt.mul` lower to rank-generic nested `scf.for` loops using the corresponding `arith` ops.
- `xt.exp` lowers to rank-generic nested `scf.for` loops using `math.exp`.
- Unary math ops lower through `math` ops or simple compositions (`sigmoid`, `silu`, `reciprocal`, `rsqrt`) over rank-generic loop nests.
- `xt.matmul` lowers to rank-2 nested loops with explicit reduction.
- `xt.mma` lowers to rank-2 nested loops with explicit reduction and type promotion from `i8` inputs into the accumulator/result element type.

The pass pipeline target is: `xt` -> `scf` / `tensor` / `memref` / `arith` / `math`, followed by standard cleanups such as `canonicalize` and `cse`.

## Tooling

`xt-opt` will register the `xt` dialect, dependent standard dialects, and the lowering/canonicalization passes. The tool must parse the sample IR from [xtile.md](/home/sjjeong94/projects/xt/xtile.md) and lower it without any remaining `xt` operations.

## Testing

lit tests will cover:

- Parser/printer round-trip for 1D, 2D, and 3D sample IR
- Verifier failures for invalid tile rank, coordinate-count mismatch, tile shapes, and mismatched types
- Canonicalization of `xt.add` with zero tensors
- Lowering output that removes all `xt` ops in favor of standard dialects
- Matmul/mma verification and lowering tests

## Notes

This repository is not currently a git repository, so the design document cannot be committed here. The implementation proceeds with local files only.
