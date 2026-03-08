//===-- Passes.td - reconcile-ffi-externs pass defs --------*- tablegen -*-===//
//
// Part of the Fidelity Framework
// Licensed under the Apache License v2.0
//
//===----------------------------------------------------------------------===//
//
// Pass declarations for FFI extern reconciliation.
//
// The Fidelity compiler's MiddleEnd (Alex) emits FFI extern declarations with
// a "ffi." prefix (e.g., @ffi.malloc, @ffi.free) to avoid symbol collisions
// with MLIR infrastructure declarations. The standard pass
// finalize-memref-to-llvm creates its own @malloc(i64) -> !llvm.ptr for
// memref.alloc lowering — if our extern used the same name, MLIR would rename
// one of them (producing @malloc_0, @malloc_1, etc.) and the linker would fail.
//
// The "ffi." prefix is a namespace convention: "these are FFI boundary calls,
// not internal MLIR infrastructure." This pass strips the prefix after standard
// lowering is complete, reconciling any type differences with the infrastructure
// declarations via explicit LLVM casts (ptrtoint, inttoptr).
//
//===----------------------------------------------------------------------===//

#ifndef RECONCILE_FFI_EXTERNS_PASSES
#define RECONCILE_FFI_EXTERNS_PASSES

include "mlir/Pass/PassBase.td"

def ReconcileFFIExterns : Pass<"reconcile-ffi-externs", "::mlir::ModuleOp"> {
  let summary = "Reconcile ffi.* extern declarations with MLIR infrastructure";
  let description = [{
    Strips the "ffi." prefix from FFI extern function declarations and call
    sites, reconciling type differences with any same-named MLIR infrastructure
    declarations via explicit LLVM dialect casts.

    Two cases are handled:

    1. **No collision** (majority of externs): The "ffi." prefix is simply
       removed. @ffi.wl_display_connect becomes @wl_display_connect. No type
       changes needed.

    2. **Collision with MLIR infrastructure** (malloc, free, aligned_alloc):
       The infrastructure declaration uses !llvm.ptr for pointer types, while
       our declaration uses i64. The pass rewrites call sites to use the
       infrastructure declaration's signature, inserting llvm.ptrtoint /
       llvm.inttoptr casts at the boundary.

    Pipeline position: after all standard dialect conversions (including
    convert-func-to-llvm), alongside resolve-closure-casts, before
    reconcile-unrealized-casts.

    Example pipeline:
      --finalize-memref-to-llvm --convert-func-to-llvm --convert-arith-to-llvm \
      --resolve-closure-casts --reconcile-ffi-externs \
      --reconcile-unrealized-casts --canonicalize
  }];

  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];

  let statistics = [
    Statistic<"numRenamed", "ffi-renamed",
              "Number of ffi.* functions simply renamed (no collision)">,
    Statistic<"numReconciled", "ffi-reconciled",
              "Number of ffi.* functions reconciled with infrastructure decls">
  ];
}

#endif // RECONCILE_FFI_EXTERNS_PASSES
