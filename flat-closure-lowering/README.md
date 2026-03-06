# flat-closure-lowering

MLIR pass plugin that resolves `unrealized_conversion_cast` operations arising from flat closure representation in the Fidelity Framework.

## Background

The Fidelity compiler's MiddleEnd (Alex) represents closures as pairs of `(code_pointer, environment_pointer)` stored as `index` values in `memref`. This uses standard MLIR dialects (func, memref, arith) and is fully target-agnostic.

Storing a function pointer as data requires type conversions that have no standard MLIR lowering:

| MiddleEnd Cast | After Standard Lowering | Resolution |
|---|---|---|
| `func_type → index` | `!llvm.ptr → i64` | `llvm.ptrtoint` |
| `index → func_type` | `i64 → !llvm.ptr` | `llvm.inttoptr` |
| `index → memref<?xi8>` | `i64 → memref descriptor` | `inttoptr` + struct build |

Alex emits these as `builtin.unrealized_conversion_cast` — the correct MLIR mechanism for deferred type resolution. Standard passes (`--convert-func-to-llvm`, etc.) convert surrounding ops but leave these casts unresolved with intermediate "ghost" function types that prevent `--reconcile-unrealized-casts` from collapsing them.

## Pass: `--resolve-closure-casts`

Identifies cast chains by their net type conversion and replaces them with proper LLVM dialect operations.

### Pipeline Position

Runs **after** all standard dialect conversions, **before** `--reconcile-unrealized-casts`:

```
mlir-opt input.mlir \
  --expand-strided-metadata \
  --memref-expand \
  --finalize-memref-to-llvm \
  --convert-vector-to-llvm \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-index-to-llvm \
  --convert-func-to-llvm \
  --convert-arith-to-llvm \
  --load-pass-plugin=flat-closure-lowering.so \
  --resolve-closure-casts \
  --reconcile-unrealized-casts \
  --canonicalize
```

### Statistics

The pass reports resolution counts via `--mlir-pass-statistics`:

- `ptr-to-int-resolved` — function address stored as integer
- `int-to-ptr-resolved` — integer recovered as function pointer
- `int-to-descriptor-resolved` — raw pointer reconstructed as memref

## Building

```sh
# From the mlir-plugins root
cmake -B build -G Ninja
cmake --build build
```

Produces `build/flat-closure-lowering/flat-closure-lowering.so`.

## Testing

```sh
# Validate a closure pattern through the full pipeline
mlir-opt test/closure_pair.mlir \
  --convert-func-to-llvm --convert-index-to-llvm --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --load-pass-plugin=build/flat-closure-lowering/flat-closure-lowering.so \
  --resolve-closure-casts \
  --reconcile-unrealized-casts \
  --canonicalize | \
mlir-translate --mlir-to-llvmir
```
