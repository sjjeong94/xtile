# Nova Dialect Design

**Goal:** Add a minimal `nova` MLIR dialect and an `xtile -> nova` conversion path that rewrites `xt.add`, `xt.mul`, and `xt.sub` into hardware-oriented `nova` ops.

**Context:** The repository already models tile programs in the `xt` dialect and can build those programs from the Python DSL. `nova.md` defines an initial hardware-facing layer that distinguishes pure elementwise execution from broadcasted execution, while keeping the rest of the program in `xt` for now.

## Scope

- Add a new `nova` dialect with two operations:
  - `nova.elementwise`
  - `nova.broadcast`
- Both ops take two tensor operands and return one tensor result.
- Both ops carry the same attributes:
  - `mode`: `i32`, where `1=add`, `2=mul`, `3=sub`
  - `lhs_s`: `f32`
  - `lhs_b`: `f32`
  - `rhs_s`: `f32`
  - `rhs_b`: `f32`
- Initial conversion only rewrites `xt.add`, `xt.mul`, and `xt.sub`.
- All other ops remain unchanged.

## Conversion Rules

- Determine the `mode` from the source op:
  - `xt.add` -> `1`
  - `xt.mul` -> `2`
  - `xt.sub` -> `3`
- Inspect operand and result tensor shapes.
- If both operand shapes exactly match the result shape, rewrite to `nova.elementwise`.
- Otherwise, if the op is broadcast-compatible under existing `xt` rules, rewrite to `nova.broadcast`.
- Preserve operand order and result type.
- Materialize default scale/bias attributes on every generated nova op:
  - `lhs_s = 1.0 : f32`
  - `lhs_b = 0.0 : f32`
  - `rhs_s = 1.0 : f32`
  - `rhs_b = 0.0 : f32`

## Architecture

- Define the `nova` dialect in parallel with `xt` using ODS plus small C++ registration glue.
- Add an `xt-to-nova` rewrite pass that runs on `func.func` and rewrites eligible `xt` binary ops to nova ops.
- Register the `nova` dialect and the new pass in `xt-opt`.
- Keep Python `xt.convert(...)` unchanged; add a separate helper that runs the `xt-to-nova` pass over a converted module.

## Python Surface

- Add `xtile.to_nova(module)` returning the same module after applying the pass pipeline.
- The helper should run `builtin.module(func.func(xt-to-nova))` via MLIR Python `PassManager`.
- This preserves a clean split:
  - `xt.convert(...)` builds frontend IR
  - `xt.to_nova(...)` applies backend-oriented conversion

## Testing

- Add a lit test covering:
  - broadcast rewrite to `nova.broadcast`
  - pure elementwise rewrite to `nova.elementwise`
  - correct `mode` values and default attributes
- Add Python tests covering:
  - broadcasted subtraction converts to `nova.broadcast` with `mode = 3`
  - elementwise multiplication converts to `nova.elementwise` with `mode = 2`
  - `xt.convert(...)` output remains in `xt` before `xt.to_nova(...)`

## Error Handling

- The pass should only match ranked tensor binary ops already verified by `xt`.
- Non-target ops are left untouched.
- No extra verifier beyond operand/result tensor consistency is required for the initial nova ops.
