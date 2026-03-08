/// reconcile-ffi-externs — Strip ffi.* prefix and reconcile with MLIR infrastructure
///
/// Part of the Fidelity Framework
/// Licensed under the Apache License v2.0
///
/// See Passes.td for pass documentation and pipeline positioning.

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Pass definition (from TableGen)
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_DEF_RECONCILEFFIEXTERNS
#include "Passes.h.inc"

//===----------------------------------------------------------------------===//
// ReconcileFFIExternsPass
//===----------------------------------------------------------------------===//

struct ReconcileFFIExternsPass
    : impl::ReconcileFFIExternsBase<ReconcileFFIExternsPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Phase 1: Index all function declarations.
    // Separate ffi.* functions from potential infrastructure counterparts.
    llvm::StringMap<LLVM::LLVMFuncOp> ffiFuncs;
    llvm::StringMap<LLVM::LLVMFuncOp> baseFuncs;

    module.walk([&](LLVM::LLVMFuncOp funcOp) {
      StringRef name = funcOp.getName();
      if (name.starts_with("ffi.")) {
        ffiFuncs[name] = funcOp;
        // Also record the base name if an infrastructure version exists
        StringRef baseName = name.drop_front(4);
        // Don't overwrite — we want the infrastructure version
        if (!baseFuncs.count(baseName)) {
          // Will be filled if found below
        }
      }
    });

    // Second walk to find infrastructure counterparts
    module.walk([&](LLVM::LLVMFuncOp funcOp) {
      StringRef name = funcOp.getName();
      if (!name.starts_with("ffi.")) {
        baseFuncs[name] = funcOp;
      }
    });

    // Phase 2: Process each ffi.* function
    for (auto &entry : ffiFuncs) {
      StringRef ffiName = entry.first();
      LLVM::LLVMFuncOp ffiFunc = entry.second;
      StringRef baseName = ffiName.drop_front(4); // strip "ffi."

      auto baseIt = baseFuncs.find(baseName);
      if (baseIt != baseFuncs.end()) {
        // Collision case: infrastructure already declared this symbol.
        // Reconcile by rewriting call sites to use the infrastructure
        // declaration, inserting type casts at the boundary.
        reconcileWithInfrastructure(module, ffiFunc, baseIt->second);
        ++numReconciled;
      } else {
        // No collision: simple rename — strip the ffi. prefix.
        renameFunction(module, ffiFunc, baseName);
        ++numRenamed;
      }
    }
  }

private:
  /// Simple rename: update the function name and all call sites.
  void renameFunction(ModuleOp module, LLVM::LLVMFuncOp ffiFunc,
                      StringRef newName) {
    StringRef oldName = ffiFunc.getName();

    // Rename the declaration
    ffiFunc.setName(newName);

    // Update all call sites
    module.walk([&](LLVM::CallOp callOp) {
      auto callee = callOp.getCallee();
      if (callee && *callee == oldName) {
        callOp.setCallee(newName);
      }
    });
  }

  /// Reconcile an ffi.* function with an infrastructure declaration that has
  /// a different type signature. Rewrites call sites to use the infrastructure
  /// declaration, inserting llvm.ptrtoint / llvm.inttoptr casts as needed.
  ///
  /// Common patterns:
  ///   @ffi.malloc(i64) -> i64     vs  @malloc(i64) -> !llvm.ptr
  ///   @ffi.free(i64)              vs  @free(!llvm.ptr)
  ///   @ffi.aligned_alloc(i64,i64) vs  @aligned_alloc(i64,i64) -> !llvm.ptr
  void reconcileWithInfrastructure(ModuleOp module, LLVM::LLVMFuncOp ffiFunc,
                                   LLVM::LLVMFuncOp baseFunc) {
    StringRef ffiName = ffiFunc.getName();
    StringRef baseName = baseFunc.getName();

    auto ffiType = ffiFunc.getFunctionType();
    auto baseType = baseFunc.getFunctionType();

    // Collect call sites first to avoid iterator invalidation
    SmallVector<LLVM::CallOp> callSites;
    module.walk([&](LLVM::CallOp callOp) {
      auto callee = callOp.getCallee();
      if (callee && *callee == ffiName) {
        callSites.push_back(callOp);
      }
    });

    // Rewrite each call site
    for (auto callOp : callSites) {
      OpBuilder builder(callOp);
      Location loc = callOp.getLoc();

      // Marshal arguments: ffi type → infrastructure type
      SmallVector<Value> newArgs;
      for (unsigned i = 0; i < callOp.getNumOperands(); ++i) {
        Value arg = callOp.getOperand(i);
        Type ffiArgTy = ffiType.getParamType(i);
        Type baseArgTy = baseType.getParamType(i);

        if (ffiArgTy != baseArgTy) {
          if (isa<IntegerType>(ffiArgTy) &&
              isa<LLVM::LLVMPointerType>(baseArgTy)) {
            // i64 → !llvm.ptr (e.g., free's pointer argument)
            arg = builder.create<LLVM::IntToPtrOp>(loc, baseArgTy, arg);
          } else if (isa<LLVM::LLVMPointerType>(ffiArgTy) &&
                     isa<IntegerType>(baseArgTy)) {
            // !llvm.ptr → i64
            arg = builder.create<LLVM::PtrToIntOp>(loc, baseArgTy, arg);
          }
        }
        newArgs.push_back(arg);
      }

      // Create new call using infrastructure declaration.
      // LLVMFunctionType: void returns have getReturnType() == LLVMVoidType.
      Type baseRetTy = baseType.getReturnType();
      bool isVoidReturn = isa<LLVM::LLVMVoidType>(baseRetTy);

      if (!isVoidReturn) {
        auto newCall = builder.create<LLVM::CallOp>(
            loc, baseRetTy, baseName, newArgs);
        Value result = newCall.getResult();

        // Marshal return: infrastructure type → ffi type
        if (callOp.getNumResults() > 0) {
          Type ffiRetTy = ffiType.getReturnType();

          if (ffiRetTy != baseRetTy) {
            if (isa<LLVM::LLVMPointerType>(baseRetTy) &&
                isa<IntegerType>(ffiRetTy)) {
              // !llvm.ptr → i64 (e.g., malloc returns pointer, we want integer)
              result = builder.create<LLVM::PtrToIntOp>(loc, ffiRetTy, result);
            } else if (isa<IntegerType>(baseRetTy) &&
                       isa<LLVM::LLVMPointerType>(ffiRetTy)) {
              // i64 → !llvm.ptr
              result = builder.create<LLVM::IntToPtrOp>(loc, ffiRetTy, result);
            }
          }
          callOp.getResult().replaceAllUsesWith(result);
        }
      } else {
        // Void return — just call
        builder.create<LLVM::CallOp>(loc, TypeRange{}, baseName, newArgs);
      }

      callOp.erase();
    }

    // Remove the ffi.* declaration — infrastructure version remains
    ffiFunc.erase();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Plugin entry point
//===----------------------------------------------------------------------===//

extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "ReconcileFFIExterns", "0.1.0", []() {
            PassRegistration<ReconcileFFIExternsPass>();
          }};
}
