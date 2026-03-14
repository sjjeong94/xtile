# CLI Xt-To-Nova Design

**Goal:** Add an `--xt-to-nova` option to the `xtile` Python CLI so users can print nova-converted MLIR directly from a kernel source file.

## Scope

- Extend `python/xtile/__main__.py`
- Keep existing CLI behavior unchanged when the flag is absent
- Allow the new option to be combined with `--canonicalize`

## Behavior

- Existing command remains:
  - `xtile softmax.py`
- New command becomes valid:
  - `xtile softmax.py --xt-to-nova`
- Per kernel, the CLI pipeline becomes:
  - `xt.convert(fn)`
  - optional `xt.to_nova(module)` when `--xt-to-nova` is set
  - optional canonicalization when `--canonicalize` is set
  - print final module text

## Option Interaction

- `--xt-to-nova` alone prints nova ops without canonicalization
- `--canonicalize` alone keeps current behavior
- `--xt-to-nova --canonicalize` first converts to nova, then canonicalizes

## Testing

- Add CLI regression coverage for:
  - `--xt-to-nova` rewrites broadcasted softmax arithmetic to `nova.broadcast`
  - default CLI output still contains `xt.sub`
  - `--xt-to-nova` and `--canonicalize` can be combined successfully
