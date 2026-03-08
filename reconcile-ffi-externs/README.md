# reconcile-ffi-externs

MLIR pass plugin that reconciles FFI extern declarations with MLIR infrastructure declarations in the Fidelity Framework.

## Background

The Fidelity compiler (Composer) emits FFI extern calls with a `ffi.` prefix in MLIR — for example, `@ffi.malloc` instead of `@malloc`. This prevents symbol collisions with declarations that MLIR's standard lowering passes create internally.

The specific collision: `finalize-memref-to-llvm` lowers `memref.alloc` operations into calls to `@malloc(i64) -> !llvm.ptr`. If the compiler also emitted `@malloc(i64) -> i64` for the user-facing `Fidelity.Libc.Memory.malloc` binding, MLIR would see two declarations with the same name but different signatures and rename one of them (producing `@malloc_0`, `@malloc_1`, etc.). The linker would then fail with "undefined reference to `malloc_42`" because no such symbol exists in libc.

The `ffi.` prefix is a namespace convention that embodies "good fences make good neighbors":
- **Inside the fence** — MLIR's `@malloc` serves `memref.alloc` lowering (internal allocator)
- **Outside the fence** — `@ffi.malloc` is the FFI boundary call to C's `malloc`

Both ultimately resolve to the same libc `malloc` at link time. This pass reconciles them after standard lowering is complete.

## Pass: `--reconcile-ffi-externs`

Processes all `ffi.*` function declarations in two modes:

| Case | Frequency | Action |
|---|---|---|
| No collision (most externs) | ~95% | Simple rename: strip `ffi.` prefix |
| Collision with infrastructure | ~5% (malloc, free, aligned_alloc) | Rewrite calls to use infrastructure signature + type casts |

### Collision Reconciliation

When an infrastructure declaration exists with a different signature, the pass rewrites call sites:

| FFI Declaration | Infrastructure Declaration | Resolution |
|---|---|---|
| `@ffi.malloc(i64) -> i64` | `@malloc(i64) -> !llvm.ptr` | Call `@malloc`, `ptrtoint` result |
| `@ffi.free(i64)` | `@free(!llvm.ptr)` | `inttoptr` arg, call `@free` |
| `@ffi.aligned_alloc(i64,i64) -> i64` | `@aligned_alloc(i64,i64) -> !llvm.ptr` | Call `@aligned_alloc`, `ptrtoint` result |

The type casts are explicit LLVM dialect ops — fully visible in IR dumps for debugging.

### Pipeline Position

Runs **after** all standard dialect conversions (where infrastructure declarations are created), alongside `--resolve-closure-casts`, **before** `--reconcile-unrealized-casts`:

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
  --load-pass-plugin=reconcile-ffi-externs.so \
  --reconcile-ffi-externs \
  --load-pass-plugin=flat-closure-lowering.so \
  --resolve-closure-casts \
  --reconcile-unrealized-casts \
  --canonicalize
```

### Statistics

The pass reports counts via `--mlir-pass-statistics`:

- `ffi-renamed` — functions simply renamed (no collision)
- `ffi-reconciled` — functions reconciled with infrastructure declarations

## Building

```sh
# From the mlir-plugins root
cmake -B build -G Ninja
cmake --build build
```

Produces `build/reconcile-ffi-externs/reconcile-ffi-externs.so`.

## Relationship to flat-closure-lowering

Both plugins operate at the same pipeline stage (post-standard-lowering, pre-reconcile-unrealized-casts) but address orthogonal concerns:

- **flat-closure-lowering**: Resolves type casts from flat closure representation (function pointers stored as data)
- **reconcile-ffi-externs**: Reconciles FFI extern symbol naming to avoid collisions with MLIR infrastructure

Neither depends on the other. Both can be promoted to higher layers of the lowering path if cleaner solutions emerge.
