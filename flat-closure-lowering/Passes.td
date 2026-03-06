//===-- Passes.td - flat-closure-lowering pass defs --------*- tablegen -*-===//
//
// Part of the Fidelity Framework
// Licensed under the Apache License v2.0
//
//===----------------------------------------------------------------------===//
//
// Pass declarations for flat closure lowering.
//
// The Fidelity Framework's MiddleEnd (Alex) emits flat closures using standard
// MLIR dialects (func, memref, arith). Closures are represented as pairs of
// (code_pointer, environment_pointer) stored as index values in memref.
//
// This requires type conversions with no standard MLIR lowering path:
//
//   func_type → index    Store function address as data
//   index → func_type    Recover function for indirect call
//   index → memref       Reconstruct memref from captured pointer
//
// Alex emits these as builtin.unrealized_conversion_cast — the correct
// target-agnostic representation. After standard dialect lowering, they become
// !llvm.ptr ↔ i64 and i64 → memref descriptor conversions that require
// explicit LLVM ops (ptrtoint, inttoptr, descriptor construction).
//
//===----------------------------------------------------------------------===//

#ifndef FLAT_CLOSURE_LOWERING_PASSES
#define FLAT_CLOSURE_LOWERING_PASSES

include "mlir/Pass/PassBase.td"

def ResolveClosureCasts : Pass<"resolve-closure-casts", "::mlir::ModuleOp"> {
  let summary = "Resolve unrealized_conversion_cast ops from flat closure IR";
  let description = [{
    Replaces unrealized_conversion_cast chains that arise from flat closure
    representation with proper LLVM dialect operations.

    After standard MLIR lowering (--convert-func-to-llvm, --finalize-memref-to-llvm,
    etc.), closure-related casts produce chains with intermediate "ghost" types
    that prevent --reconcile-unrealized-casts from collapsing them.

    This pass identifies three patterns by their net type conversion:

      !llvm.ptr → i64          →  llvm.ptrtoint   (function address as data)
      i64 → !llvm.ptr          →  llvm.inttoptr   (data as function pointer)
      i64 → memref descriptor  →  inttoptr + descriptor construction

    Pipeline position: after all standard dialect conversions, before
    --reconcile-unrealized-casts.

    Example pipeline:
      --finalize-memref-to-llvm --convert-func-to-llvm --convert-arith-to-llvm \
      --resolve-closure-casts \
      --reconcile-unrealized-casts --canonicalize
  }];

  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];

  let statistics = [
    Statistic<"numPtrToInt", "ptr-to-int-resolved",
              "Number of ptr→int cast chains resolved">,
    Statistic<"numIntToPtr", "int-to-ptr-resolved",
              "Number of int→ptr cast chains resolved">,
    Statistic<"numIntToDescriptor", "int-to-descriptor-resolved",
              "Number of int→memref descriptor cast chains resolved">
  ];
}

#endif // FLAT_CLOSURE_LOWERING_PASSES
