# Fidelity MLIR Plugins

MLIR pass plugins for the [Fidelity Framework](https://github.com/FidelityFramework). These plugins extend the MLIR lowering pipeline with transformations required by the Fidelity compilation model.

Each plugin is a shared library loaded via `mlir-opt --load-pass-plugin=<plugin>.so`.

## Plugins

| Plugin | Pass | Purpose |
|---|---|---|
| [flat-closure-lowering](flat-closure-lowering/) | `--resolve-closure-casts` | Resolve closure-related type casts during LLVM lowering |
| [reconcile-ffi-externs](reconcile-ffi-externs/) | `--reconcile-ffi-externs` | Reconcile FFI extern declarations with MLIR infrastructure |

### flat-closure-lowering

Resolves closure-related type casts during LLVM lowering. The Fidelity compiler represents closures as flat structures containing captured values and a function pointer. During lowering to the LLVM dialect, type mismatches arise between the generic closure representation and the concrete function signatures. This pass resolves those casts so that `--reconcile-unrealized-casts` can complete cleanly.

**Pass flag**: `--resolve-closure-casts`

### reconcile-ffi-externs

Reconciles FFI extern declarations with infrastructure declarations that MLIR's standard lowering passes generate internally. The Fidelity compiler emits FFI calls with a `ffi.` prefix (e.g., `@ffi.malloc`) to avoid symbol collisions with declarations that passes like `finalize-memref-to-llvm` create (e.g., `@malloc`). After standard lowering is complete, this pass strips the `ffi.` prefix from non-colliding declarations and rewrites call sites for colliding declarations to use the infrastructure signature with explicit `ptrtoint`/`inttoptr` casts.

**Pass flag**: `--reconcile-ffi-externs`

## Building

Requires MLIR/LLVM development headers (tested with LLVM 21).

```sh
cmake -B build -G Ninja
cmake --build build
```

The built plugins are placed in `build/<plugin-name>/`.

## Usage

```sh
mlir-opt input.mlir \
  --load-pass-plugin=build/reconcile-ffi-externs/reconcile-ffi-externs.so \
  --reconcile-ffi-externs \
  --load-pass-plugin=build/flat-closure-lowering/flat-closure-lowering.so \
  --resolve-closure-casts
```

## Architecture

These plugins operate exclusively in the **BackEnd** of the Composer compilation pipeline. The MiddleEnd (Alex) emits target-agnostic MLIR using standard dialects. Plugins bridge the semantic gap between that representation and target-specific lowering requirements.

Plugins do not modify or depend on the MiddleEnd. They are loaded by the BackEnd's `mlir-opt` invocation at the appropriate pipeline position.

## Known Issues

### Arch Linux: MLIR cmake version mismatch

On Arch Linux, the `mlir` package may ship an `MLIRConfig.cmake` that hardcodes a stale LLVM version (e.g. `21.1.6` when LLVM `21.1.8` is installed). This causes cmake to fail with a version mismatch when `MLIRConfig.cmake` calls `find_package(LLVM 21.1.6)` against a newer LLVM.

**Diagnosis**: The error will mention an LLVM version that doesn't match your installed version:

```
Could not find a configuration file for package "LLVM" that is compatible
with requested version "21.1.6".
```

**Fix**: Correct the hardcoded version in `MLIRConfig.cmake`:

```sh
sudo sed -i 's/set(LLVM_VERSION 21.1.6)/set(LLVM_VERSION 21.1.8)/' /usr/lib/cmake/mlir/MLIRConfig.cmake
```

Adjust the version strings to match your actual installed LLVM version (`llvm-config --version`). This is a packaging bug — the MLIR and LLVM packages should agree on the version string.

## License

Apache License 2.0
