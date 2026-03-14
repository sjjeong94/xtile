# Install MLIR Script Design

**Goal:** Replace the one-shot `scripts/install_mlir.sh` with a reusable repository-local installer that can clone, reuse, configure, build, and install MLIR under `../llvm-project`.

## Scope

- Keep the default source root at `../llvm-project`
- Keep the default LLVM/MLIR ref at `llvmorg-20.1.8`
- Make repeated execution safe
- Remove distro-specific package installation from the script

## Behavior

- Resolve `REPO_ROOT` from the script location
- Resolve default paths:
  - source root: `"$REPO_ROOT/../llvm-project"`
  - build dir: `"$LLVM_ROOT/build"`
  - install dir: `"$LLVM_ROOT/install"`
- Allow overrides:
  - `XTILE_LLVM_ROOT`
  - `LLVM_REF`
  - `XTILE_LLVM_BUILD_DIR`
  - `XTILE_LLVM_INSTALL_DIR`
- Validate required tools:
  - `git`
  - `cmake`
- Prefer Ninja if available, otherwise fall back to the default generator
- If source root does not exist, clone `https://github.com/llvm/llvm-project.git`
- If source root exists, verify it is a git repository and reuse it
- Fetch the target ref when needed, then `git checkout` it
- Re-run `cmake -S/-B` safely on every invocation
- Re-run `cmake --build ... --target install` safely on every invocation

## Error Handling

- Fail with a clear message if an existing `XTILE_LLVM_ROOT` is not a git repo
- Fail with a clear message if required tools are missing
- Avoid implicit `sudo` or package manager actions

## Validation

- `bash -n scripts/install_mlir.sh` must pass
- Running the script should not fail just because `../llvm-project` already exists
