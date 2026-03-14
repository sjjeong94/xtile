# Python CLI Nova Optimize Design

## Goal

Add a `--nova-optimize` option to the Python CLI so Nova peephole optimizations can be run from `xtile` after `--xt-to-nova`.

## Scope

- Add a new CLI flag: `--nova-optimize`
- Run the existing MLIR `nova-optimize` pass through Python MLIR bindings
- Keep the pipeline order explicit:
  - `convert`
  - optional `xt.to_nova`
  - optional `nova-optimize`
  - optional `canonicalize`

## Behavior

- `--xt-to-nova --nova-optimize` should produce folded Nova IR
- `--nova-optimize` without `--xt-to-nova` should still succeed; it simply runs the pass over a module that may not contain Nova ops
- `--canonicalize` remains the last optional step

## Testing

Add CLI regression coverage for:

- `--xt-to-nova --nova-optimize` folding `nova.scalar` into `nova.broadcast`/`nova.elementwise`
- `--xt-to-nova --nova-optimize --canonicalize` continuing to work
