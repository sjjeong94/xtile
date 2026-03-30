# xtile Programming Model

xtile is a target-independent programming model for expressing tile-level data movement, layout transformation, and computation.

The main goal of xtile is to let users describe **what each tile-level block should do**, while leaving hardware-specific scheduling, memory materialization, and synchronization to the compiler and lower target dialects.

In short, xtile is designed to be:

- **tile-centric**
- **block-oriented**
- **target-independent**
- **compiler-managed for execution details**

---

## Why xtile?

Many programming models are either:

- too low-level, exposing hardware execution details directly, or
- too high-level, hiding tile structure and data movement completely

xtile aims for the middle ground.

It gives users explicit control over:

- how a task is partitioned into blocks
- which tile each block operates on
- how tiles are loaded, computed, and stored
- which tiles are logically shared across blocks

At the same time, xtile does **not** require users to manage:

- physical scheduling
- hardware execution unit mapping
- shared memory allocation/free
- explicit synchronization such as barriers or waits

Those are compiler responsibilities.

---

## Programming Model Overview

xtile programs are written in terms of **Task**, **Grid**, **Block**, and **Kernel**.

### Task
A **Task** is the full computation the user wants to perform.

For example, adding two `M x N` matrices is one Task.

### Block
A **Block** is the basic unit of work.

A Task is divided into multiple Blocks, and each Block is responsible for a subset of the overall work.

### Grid
A **Grid** is the collection of all Blocks.

A Grid can be up to 3-dimensional.

### Kernel
A **Kernel** is the program written from the perspective of **one Block**.

The same Kernel is applied to every Block in the Grid, and each Block uses its own block id to determine which tile it should process.

---

## Block-Centric Execution

In xtile, the user writes code as if describing the work of a single Block.

Inside the Kernel, the current Block’s coordinate can be obtained using:

```python
bid0 = xt.bid(axis=0)
bid1 = xt.bid(axis=1)
bid2 = xt.bid(axis=2)
```

These values identify the current Block within the Grid.

The user can use them to compute which tile the Block should load and process.

### Important note
Blocks are **logically parallel**.

This means xtile assumes that each Block represents an independent unit of work.  
However, whether Blocks are actually executed in parallel, partially serialized, or remapped onto hardware resources is determined by the compiler and hardware backend.

---

## Memory Model

xtile defines three memory classes:

- **Global Memory**
- **Shared Memory**
- **Local Memory**

### Global Memory
Global Memory is visible to the entire Grid.

Data in Global Memory is represented as an **Array**.

An Array has:

- `shape`
- `dtype`
- `stride`

Arrays are mutable.

### Local Memory
Local Memory is private to a Block.

Data loaded by default from Global Memory becomes a **Tile** in Local Memory.

### Shared Memory
Shared Memory is a logical memory class used when multiple Blocks reuse the same tile data.

In xtile, Shared Memory is **not** treated as explicitly user-managed physical scratchpad memory.  
Instead, it represents the fact that a tile is logically shared across a group of Blocks.

The compiler decides how that shared tile is actually materialized.

---

## Data Types: Array vs Tile

xtile distinguishes between two major data objects.

### Array
An **Array** lives in Global Memory.

Typical examples include:

- input tensors
- output tensors
- weights
- large global buffers

### Tile
A **Tile** lives in Local Memory or Shared Memory.

A Tile has:

- `shape`
- `dtype`

Tiles are the basic unit of computation in xtile.

A typical xtile program does the following:

1. load a Tile from an Array
2. compute on the Tile
3. store the resulting Tile back into an Array

---

## Tile Coordinates

One of the key design choices in xtile is that the `index` argument of `xt.load` and `xt.store` is a **tile coordinate**, not an element coordinate.

For example:

```python
tile = xt.load(a, index=(0, 1), shape=(32, 64))
```

This does **not** mean loading the element at row 0, column 1.

Instead, it means:

- partition `a` logically into tiles of shape `(32, 64)`
- load the tile at tile coordinate `(0, 1)`

So xtile programs operate directly in terms of **which tile** is accessed, rather than manually computing element offsets.

This makes the model naturally tile-oriented.

---

## Load and Store

### Load
A tile can be loaded from an Array using `xt.load`.

```python
tile = xt.load(a, index=(bid0, bid1), shape=(32, 64))
```

By default, this loads a tile from Global Memory into Local Memory.

### Store
A tile can be written back to an Array using `xt.store`.

```python
xt.store(c, index=(bid0, bid1), shape=(32, 64), tile=tile)
```

This stores the Tile into the target Array at the given tile coordinate.

---

## Shared Tiles

xtile allows a Tile to be marked as shared across a group of Blocks.

For example:

```python
tile = xt.load(a, index=(0, 1), shape=(32, 64), shared=1)
```

This indicates that the loaded tile is a **Shared Tile** rather than a Local Tile.

### Meaning of `shared=k`
`shared=k` defines a sharing group based on Block IDs.

Blocks belong to the same sharing group if their Block IDs are equal for all axes from `k` to the last Grid axis.

For a 3D Grid with Block ID `(b0, b1, b2)`:

- `shared=1`  
  blocks with the same `b1` and `b2` share the same tile
- `shared=2`  
  blocks with the same `b2` share the same tile

This provides a structured way to describe tile reuse across Blocks.

---

## Compiler-Managed Shared Memory

A key principle of xtile is that Shared Memory is **logical**, not manually managed by the user.

The user specifies that a tile is shared.

The compiler is responsible for deciding:

- where the shared tile should live
- how long it should live
- how it should be reused
- when synchronization is necessary
- how it maps to target-specific memory resources

This means users do **not** write:

- explicit allocation
- explicit free
- explicit barrier
- explicit wait
- explicit synchronization logic

Instead, the compiler derives these from:

- dataflow
- liveness
- block sharing structure
- dependency analysis

---

## Dependency Model

xtile v0.1 does not expose explicit asynchronous primitives such as:

- tokens
- events
- barriers
- waits

Execution dependencies are inferred from program structure and data usage.

The compiler may insert synchronization as needed to preserve correct semantics.

This keeps the programming model simple while still allowing the backend to generate efficient target-specific execution behavior.

---

## Example

Below is a simple matrix addition example:

```python
def kernel(a, b, c):
    bid0 = xt.bid(axis=0)
    bid1 = xt.bid(axis=1)

    ta = xt.load(a, index=(bid0, bid1), shape=(32, 64))
    tb = xt.load(b, index=(bid0, bid1), shape=(32, 64))

    tc = ta + tb

    xt.store(c, index=(bid0, bid1), shape=(32, 64), tile=tc)
```

### What this does
- each Block processes one tile
- `bid0`, `bid1` determine which tile the Block owns
- `xt.load` reads the input tiles
- the computation is performed tile-wise
- `xt.store` writes the output tile back

This reflects the basic xtile workflow:

**Array → Tile → Compute → Tile → Array**

---

## Design Philosophy

xtile is built around a clear separation of responsibilities.

### User responsibilities
The user describes:

- how the task is partitioned
- which tile a Block accesses
- what computation happens on the tile
- whether a tile is local or shared

### Compiler responsibilities
The compiler decides:

- execution mapping
- scheduling
- memory planning
- synchronization insertion
- lowering to hardware-specific dialects

This separation is the core idea behind xtile.

---

## What xtile v0.1 Does Not Define

xtile v0.1 intentionally leaves several things out of the user-facing programming model:

- thread-level execution model
- warp/wave/subgroup semantics
- explicit barrier semantics
- explicit async token/event model
- full memory consistency model
- multi-writer shared update semantics
- physical scheduling details
- hardware-specific execution mapping

These are either compiler concerns or future extensions.

---

## Summary

xtile is a programming model for writing **tile-level block programs** in a hardware-independent way.

It allows users to express:

- task decomposition into blocks
- tile-based data access
- tile-based computation
- logical sharing of tiles across blocks

while letting the compiler handle:

- synchronization
- lifetime planning
- memory materialization
- target-specific lowering

In one sentence:

> xtile lets the user describe **what each tile-level block does**, while the compiler decides **how that program is executed on real hardware**.
