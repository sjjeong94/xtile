# XT / xtile DSL Programming Model

## 1. What xtile is

`xtile` is not a general-purpose runtime language. It is a Python-embedded tracing DSL that records a straight-line, tile-local computation and lowers it into a staged MLIR pipeline:

`Python kernel` -> `xt dialect` -> `nova dialect` -> `x1 dialect`

At the user level, you write a Python function decorated with `@xt.kernel`. During `xt.convert(...)`, that function is executed once in a special trace context. The function does not compute real tensor values. Instead, each `xt.*` call constructs symbolic tile values and appends operations to an internal trace, which is then emitted as MLIR.

The resulting programming model is therefore:

- Host Python chooses global tensor shapes, tile sizes, and grid shape.
- Kernel code describes one tile program.
- `xt.get_tile_block_id` identifies which tile instance is being described.
- `xt.load` and `xt.store` connect global memory and tile-local tensors.
- Tile-local ops form a pure SSA dataflow graph.
- Lowering passes progressively turn that graph into more hardware-oriented forms.

## 2. Core abstraction: a kernel is a tile program template

A kernel function is a template for one block/tile instance, not an imperative loop nest over all elements.

Example shape of the model:

```python
@xt.kernel
def softmax(inp, out, trow, col):
    bid_x = xt.bid(0)

    x = xt.load(inp, index=(bid_x, 0), shape=(trow, col))
    x = x - xt.max(x, axis=-1)
    x = xt.exp(x)
    r = xt.sum(x, axis=-1)
    x = x / r
    xt.store(out, index=(bid_x, 0), tile=x)
```

This means:

- one block is assigned by `bid_x`
- that block loads one tile from global memory
- all computation happens on tile tensors
- the tile result is stored back

The host supplies `grid=(gx, gy, gz)` to say how many instances of this tile program exist.

## 3. The host side vs. kernel side

There are two distinct layers.

### Host side

The host side constructs:

- `xt.Array(shape=..., dtype=...)` values that describe global buffers
- integer parameters such as tile sizes
- grid dimensions
- whether `double_buffering` is enabled

The host then calls:

```python
ir = xt.convert(kernel_fn, args=(...), grid=(...), double_buffering=...)
```

The host is responsible for tiling decisions. For example, code in `kernels/add.py` and `kernels/matmul.py` computes `grid` with `xt.cdiv(...)`.

### Kernel side

The kernel body can only use a restricted DSL:

- `xt.bid(dim)`
- `xt.load(...)`
- `xt.load_conv2d(...)`
- `xt.store(...)`
- tile ops such as `add`, `sub`, `mul`, `matmul`, `mma`, `exp`, `sum`, `max`
- shape transforms such as `reshape` and `transpose`
- `astype`

The kernel must not return a value. `xt.convert` rejects non-`None` returns.

## 4. Values and types

### Global arrays

`xt.Array` describes a ranked global tensor-like buffer:

- shape must be a non-empty tuple of positive integers
- dtype is currently one of:
  - `xt.float32`
  - `xt.int8`

When lowered into `xt` IR, arrays become `memref<...>` kernel arguments.

### Tile values

Inside the kernel, `xt.load` and all tile ops produce `TensorValue` objects.

These are symbolic SSA values with:

- static shape
- static dtype
- no concrete runtime payload during tracing

The entire DSL assumes statically shaped ranked tensors.

## 5. Execution model: tracing, not eager execution

The most important inference from the code is that xtile uses trace-time symbolic execution.

When `xt.convert(...)` runs:

1. A `TraceContext` is created.
2. Kernel parameters are replaced with symbolic objects:
   - array params become `KernelArg`
   - integer params remain compile-time integers
3. The Python kernel runs once.
4. Every `xt.*` call appends an operation node to `TraceContext.operations`.
5. The trace is serialized into a single `func.func` in MLIR.

Consequences:

- Python control flow is not the real execution model. Only the traced `xt.*` calls matter.
- Integer kernel arguments behave like compile-time constants for shaping/index construction.
- The DSL is effectively a graph builder for a straight-line tile program.

## 6. Block IDs and tile coordinates

`xt.bid(0)`, `xt.bid(1)`, and `xt.bid(2)` are the only block-id queries.

They lower to:

```mlir
%bid:3 = xt.get_tile_block_id : i32, i32, i32
```

The three components represent logical grid coordinates `(x, y, z)`.

These IDs are used as tile coordinates in `xt.load` / `xt.store`. A key semantic detail appears in the `xt -> nova` lowering: when a tile coordinate is constant, it is multiplied by the tile extent of that dimension to produce a concrete start offset. In other words, the index tuple is in **tile coordinates**, not raw element offsets.

For example:

- `xt.load(..., index=(bid_x, 0), shape=(64, 64))`

means:

- load the tile whose top-left starts at `(bid_x * 64, 0 * 64)`

not:

- load starting at `(bid_x, 0)` elements

## 7. Memory model

The model separates:

- global memory buffers: kernel arguments as memrefs
- local tile tensors: results of `xt.load` and tile ops

### Load / store

`xt.load(array, index=..., shape=..., shared=...)`

- reads a statically shaped tile from a global array
- requires the source rank, index rank, and result rank to match
- preserves element type

`xt.store(array, index=..., tile=...)`

- writes a tile tensor back to global memory
- requires destination rank, index rank, and tile rank to match

### Shared hint

`xt.load` accepts `shared=None|0|1|2`.

From `xt-serialize`, these are reuse hints:

- `shared=1`: share across the x dimension at fixed `(y, z)`
- `shared=2`: share across both x and y at fixed `z`
- `None`/`0`: no sharing

This is not an explicit user-managed shared-memory API. It is a scheduling/reuse hint consumed by serialization and later lowerings.

### Double buffering

`double_buffering=True` is stored as a function attribute:

```mlir
attributes {xt.double_buffering = 1 : i32}
```

It is part of the kernel scheduling contract rather than an operation in the DSL surface.

## 8. Computation model: pure tile dataflow

Once data is loaded, computation is expressed as pure SSA tensor transformations.

### Binary ops

- `xt.add`
- `xt.sub`
- `xt.mul`

They require:

- matching dtypes
- broadcast-compatible shapes

In Python tracing, broadcasting is currently restricted to equal-rank, rowwise-style compatibility where each rhs dimension is either equal to lhs or `1`. In the `xt` verifier, the MLIR op supports standard same-rank-or-lower-rank trailing broadcast rules.

### Unary ops

- `xt.exp`
- `xt.cos`
- `xt.sin`
- `xt.reciprocal`
- `xt.rsqrt`
- `xt.sigmoid`
- `xt.tanh`
- `xt.silu`

These preserve shape and dtype.

### Division

There is no first-class `xt.div` op in the Python DSL.

`a / b` lowers to:

```text
xt.reciprocal(b)
xt.mul(a, reciprocal_b)
```

So the programming model treats division as a derived pattern, not a primitive.

### Reductions

- `xt.sum(tile, axis=...)`
- `xt.max(tile, axis=...)`

Current Python tracing supports rank-2 tensors and replaces the reduced dimension with size `1`. Negative axes are normalized in the Python DSL.

This means reductions preserve rank, producing shapes like:

- `(M, N)` -> `(M, 1)` for `axis=1`
- `(M, N)` -> `(1, N)` for `axis=0`

### Matmul and MMA

`xt.matmul(lhs, rhs)`

- rank-2 only
- requires `lhs.shape[1] == rhs.shape[0]`
- result is `(lhs.shape[0], rhs.shape[1])`
- dtype rules:
  - same dtype in -> same dtype out
  - `int8 x int8 -> float32` in the Python DSL

`xt.mma(lhs, rhs, acc)`

- rank-2 only
- `lhs` and `rhs` must be `int8`
- accumulator must currently be `float32` in the Python DSL
- accumulator shape must match the matmul result shape

The intent is explicit matrix-multiply-accumulate on tile operands.

### Casts

`xt.astype(tile, dtype=...)` currently supports:

- `int8 -> float32` via `xt.itof`
- `float32 -> int8` via `xt.ftoi`

Same-dtype casts are elided in the Python DSL by returning the input tile unchanged.

## 9. Shape transforms

The DSL includes three non-compute structural transforms.

### Reshape

`xt.reshape(tile, shape=...)`

- preserves dtype
- requires equal element counts

### Transpose

`xt.transpose(tile)`

- currently rank-3 only in the Python DSL
- preserves dim 0
- swaps dims 1 and 2

So `(A, B, C)` becomes `(A, C, B)`.

### Permute

The `xt` dialect includes `xt.permute(input) {permutation = [...]}` and lowers it to `nova.permute`.

This is a more general dimension reorder than `transpose`. The Python surface in `xtile/dsl.py` exposes `transpose` and `reshape`, but does not currently expose a public `permute(...)` helper even though the IR supports it.

## 10. Convolution-specific load model

`xt.load_conv2d(...)` is not a generic convolution op over global buffers. It represents a fused pattern:

1. locate the input slice needed for one output tile
2. handle boundary padding
3. apply convolution with a filter tile

Surface constraints:

- source must be rank-4 `int8`
- filter must be rank-4 `int8`
- result is rank-4 and produced as `float32` in the Python DSL
- `group > 0`
- `pad` has 4 ints
- `stride` and `dilation` have 2 positive ints

The `xt -> nova` lowering makes the semantics explicit:

- it computes the exact input slice to load
- it adjusts pad values when the tile lies on image boundaries
- it emits:
  - `nova.load` of the required source slice
  - `nova.conv2d` on the loaded slice and filter tile

So `xt.load_conv2d` is really a **tile-local convolution fetch+compute primitive**.

## 11. Serialization model

The `xt` IR is still a symbolic tile program. The `xt-serialize` pass converts that into a concrete serialized schedule across the declared grid.

Requirements from the pass:

- function must have `xt.grid`
- function must be void
- function must be single-block

The pass:

- iterates over all `(x, y, z)` in the grid
- clones the kernel body once per tile instance
- replaces `xt.get_tile_block_id` results with concrete `arith.constant` values
- reuses selected shared loads according to the `shared` hint

This is a strong clue about the true model:

- the kernel body is a per-tile template
- whole-kernel execution is an explicit serialized expansion of that template

## 12. Lowering model by dialect

### `xt`: user-facing tile IR

`xt` is the clean semantic IR for:

- tile/block coordinates
- tile loads/stores
- tile tensor math
- explicit static shapes
- scheduling hints (`xt.grid`, `xt.double_buffering`, `shared`)

### `nova`: hardware-oriented tile IR

`xt_to_nova` rewrites `xt` into more implementation-oriented ops such as:

- `nova.load`
- `nova.store`
- `nova.elementwise`
- `nova.broadcast`
- `nova.scalar`
- `nova.square`
- `nova.reduce`
- `nova.matmul`
- `nova.conv2d`
- `nova.permute`

Important semantic refinements happen here:

- constant tile indices become constant element starts
- scalar constants may fold into specialized scalar ops
- `mul(x, x)` becomes `nova.square`
- `matmul` gains explicit scale/bias operands
- `load_conv2d` becomes load-plus-conv2d

### `x1`: low-level command IR

The final `x1` dialect is command-style hardware IR. By that stage, the model is no longer "a kernel DSL" but "a sequence of hardware commands" such as loads, barriers, compute, and stores.

## 13. What the language is good at

The inferred sweet spot is:

- statically shaped NPU tile kernels
- dense tensor kernels with explicit tiling
- rowwise broadcast/reduction patterns
- matmul / mma pipelines
- conv2d tiles
- code generation workflows where the user wants to control tile sizes and grid decomposition in Python

Examples in the repo fit this exactly:

- add
- softmax
- matmul with scale/bias and cast
- conv2d

## 14. What the language is not

Based on the current code, xtile is not:

- a full tensor programming language
- a dynamic-shape system
- a control-flow-heavy kernel language
- an automatic scheduler/autotiler
- a runtime that executes Python kernels directly

Notable current limitations inferred from the implementation:

- kernels are effectively straight-line
- serialization only supports single-block void functions
- Python tracing only accepts `xt.Array` and integer kernel arguments
- the Python DSL supports only a narrow dtype set
- several ops in the IR are not exposed at the Python surface
- some Python-side checks are stricter than the underlying IR

## 15. Concise mental model

The most accurate short description is:

> xtile is a Python-fronted, statically shaped, tile-program tracing DSL for NPU-style kernels. You write one block-level tile computation over symbolic tile tensors, annotate how tiles map over a 3D grid, and rely on MLIR passes to serialize, specialize, and lower that tile program into hardware-oriented command IR.

If you keep only five rules in mind, they are these:

1. `@xt.kernel` describes one tile program, not the whole tensor loop nest.
2. `xt.bid(d)` gives tile coordinates in a 3D launch grid.
3. `xt.load` / `xt.store` move between global memrefs and local tile tensors.
4. All compute is static-shape SSA over tile tensors.
5. The compilation pipeline progressively turns that symbolic tile program into concrete hardware commands.
