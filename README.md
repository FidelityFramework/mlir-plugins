# Fidelity MLIR Plugins

MLIR pass plugins for the [Fidelity Framework](https://github.com/FidelityFramework). These plugins extend the MLIR lowering pipeline with transformations required by the Fidelity compilation model.

Each plugin is a shared library loaded via `mlir-opt --load-pass-plugin=<plugin>.so`.

## Plugins

| Plugin | Pass | Purpose |
|---|---|---|
| [flat-closure-lowering](flat-closure-lowering/) | `--resolve-closure-casts` | Resolve closure-related type casts during LLVM lowering |

## Building

Requires MLIR/LLVM development headers (tested with LLVM 21).

```sh
cmake -B build -G Ninja
cmake --build build
```

The built plugins are placed in `build/flat-closure-lowering/`.

## Usage

```sh
mlir-opt input.mlir \
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
