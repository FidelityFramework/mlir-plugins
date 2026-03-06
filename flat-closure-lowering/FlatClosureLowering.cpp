/// flat-closure-lowering — Resolve closure-related unrealized_conversion_cast ops
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
#define GEN_PASS_DEF_RESOLVECLOSURECASTS
#include "Passes.h.inc"

//===----------------------------------------------------------------------===//
// Chain analysis utilities
//===----------------------------------------------------------------------===//

/// Trace through a chain of single-input unrealized_conversion_cast ops to
/// find the original source value (the input to the first cast in the chain).
static Value traceToSource(Value value) {
  while (auto castOp = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (castOp.getInputs().size() != 1)
      break;
    value = castOp.getInputs()[0];
  }
  return value;
}

/// Collect all unrealized_conversion_cast ops in a chain ending at `sinkOp`,
/// tracing backwards through single-input casts. Returns them in
/// source-to-sink order.
static SmallVector<UnrealizedConversionCastOp>
collectCastChain(UnrealizedConversionCastOp sinkOp) {
  SmallVector<UnrealizedConversionCastOp> reverse;
  auto current = sinkOp;

  reverse.push_back(current);
  while (current.getInputs().size() == 1) {
    auto prev =
        current.getInputs()[0].getDefiningOp<UnrealizedConversionCastOp>();
    if (!prev)
      break;
    reverse.push_back(prev);
    current = prev;
  }

  SmallVector<UnrealizedConversionCastOp> chain;
  for (int i = reverse.size() - 1; i >= 0; --i)
    chain.push_back(reverse[i]);
  return chain;
}

/// Erase a chain of cast ops that are now dead, from sink to source.
static void eraseCastChain(UnrealizedConversionCastOp sinkOp) {
  auto chain = collectCastChain(sinkOp);
  for (int i = chain.size() - 1; i >= 0; --i) {
    if (chain[i].use_empty())
      chain[i].erase();
  }
}

//===----------------------------------------------------------------------===//
// Pattern classification
//===----------------------------------------------------------------------===//

/// Is this the sink of a chain whose net effect is !llvm.ptr → IntegerType?
static bool isPointerToIntChain(UnrealizedConversionCastOp sinkOp) {
  Value source = traceToSource(sinkOp.getResult(0));
  return isa<LLVM::LLVMPointerType>(source.getType()) &&
         isa<IntegerType>(sinkOp.getResult(0).getType());
}

/// Is this the sink of a chain whose net effect is IntegerType → !llvm.ptr?
static bool isIntToPointerChain(UnrealizedConversionCastOp sinkOp) {
  Value source = traceToSource(sinkOp.getResult(0));
  return isa<IntegerType>(source.getType()) &&
         isa<LLVM::LLVMPointerType>(sinkOp.getResult(0).getType());
}

/// Is this the sink of a chain whose net effect is IntegerType → LLVM struct
/// (a memref descriptor)?
///
/// After --finalize-memref-to-llvm, memref types become:
///   !llvm.struct<(ptr, ptr, i64, array<N x i64>, array<N x i64>)>
/// The first two fields are allocated/aligned pointers; field [2] is offset.
static bool isIntToMemRefDescriptorChain(UnrealizedConversionCastOp sinkOp) {
  Value source = traceToSource(sinkOp.getResult(0));
  if (!isa<IntegerType>(source.getType()))
    return false;

  auto structType =
      dyn_cast<LLVM::LLVMStructType>(sinkOp.getResult(0).getType());
  if (!structType || structType.isIdentified())
    return false;

  // Memref descriptor signature: (ptr, ptr, i64, array..., array...)
  auto body = structType.getBody();
  if (body.size() < 3)
    return false;

  return isa<LLVM::LLVMPointerType>(body[0]) &&
         isa<LLVM::LLVMPointerType>(body[1]) && isa<IntegerType>(body[2]);
}

//===----------------------------------------------------------------------===//
// ResolveClosureCastsPass
//===----------------------------------------------------------------------===//

struct ResolveClosureCastsPass
    : impl::ResolveClosureCastsBase<ResolveClosureCastsPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Collect all cast ops first to avoid iterator invalidation.
    SmallVector<UnrealizedConversionCastOp> casts;
    module.walk(
        [&](UnrealizedConversionCastOp op) { casts.push_back(op); });

    for (auto castOp : casts) {
      // Skip ops erased by prior iterations.
      if (!castOp->getParentOp())
        continue;
      if (castOp.getNumResults() != 1 || castOp.getInputs().size() != 1)
        continue;

      // Only process chain sinks — the last cast whose result feeds a
      // non-cast consumer. This ensures we replace the entire chain at once.
      bool isSink = true;
      for (Operation *user : castOp.getResult(0).getUsers()) {
        if (isa<UnrealizedConversionCastOp>(user)) {
          isSink = false;
          break;
        }
      }
      if (!isSink)
        continue;

      OpBuilder builder(castOp);
      Value source = traceToSource(castOp.getResult(0));
      Location loc = castOp.getLoc();

      // Pattern 1: !llvm.ptr → i64
      // Function address stored as integer data.
      // Lowered via: llvm.ptrtoint
      if (isPointerToIntChain(castOp)) {
        auto intType = cast<IntegerType>(castOp.getResult(0).getType());
        Value result =
            builder.create<LLVM::PtrToIntOp>(loc, intType, source);
        castOp.getResult(0).replaceAllUsesWith(result);
        eraseCastChain(castOp);
        ++numPtrToInt;
        continue;
      }

      // Pattern 2: i64 → !llvm.ptr
      // Integer data recovered as function pointer for indirect call.
      // Lowered via: llvm.inttoptr
      if (isIntToPointerChain(castOp)) {
        auto ptrType =
            cast<LLVM::LLVMPointerType>(castOp.getResult(0).getType());
        Value result =
            builder.create<LLVM::IntToPtrOp>(loc, ptrType, source);
        castOp.getResult(0).replaceAllUsesWith(result);
        eraseCastChain(castOp);
        ++numIntToPtr;
        continue;
      }

      // Pattern 3: i64 → memref descriptor struct
      // Raw pointer (stored as integer) reconstructed as memref for capture
      // extraction. The MiddleEnd always follows this with
      // memref.reinterpret_cast to establish correct bounds.
      //
      // Lowered via: llvm.inttoptr + descriptor struct construction.
      if (isIntToMemRefDescriptorChain(castOp)) {
        auto structType =
            cast<LLVM::LLVMStructType>(castOp.getResult(0).getType());

        // Integer → pointer
        auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
        Value ptr = builder.create<LLVM::IntToPtrOp>(loc, ptrType, source);

        // Build memref descriptor:
        //   [0] allocated_ptr  = ptr
        //   [1] aligned_ptr    = ptr  (no alignment offset for raw captures)
        //   [2] offset         = 0
        //   [3] sizes          (sentinel for dynamic — reinterpret_cast sets actual)
        //   [4] strides        (1 for contiguous byte array)
        Value desc = builder.create<LLVM::UndefOp>(loc, structType);
        desc = builder.create<LLVM::InsertValueOp>(loc, desc, ptr,
                                                   ArrayRef<int64_t>{0});
        desc = builder.create<LLVM::InsertValueOp>(loc, desc, ptr,
                                                   ArrayRef<int64_t>{1});

        auto i64Ty = builder.getI64Type();
        Value zero = builder.create<LLVM::ConstantOp>(
            loc, i64Ty, builder.getI64IntegerAttr(0));
        desc = builder.create<LLVM::InsertValueOp>(loc, desc, zero,
                                                   ArrayRef<int64_t>{2});

        // Populate size/stride arrays based on descriptor rank
        auto body = structType.getBody();
        if (body.size() == 5) {
          // 1-D memref descriptor: sizes[0] = sentinel, strides[0] = 1
          // Actual size established by reinterpret_cast downstream.
          Value sentinel = builder.create<LLVM::ConstantOp>(
              loc, i64Ty,
              builder.getI64IntegerAttr(
                  std::numeric_limits<int64_t>::max()));
          Value one = builder.create<LLVM::ConstantOp>(
              loc, i64Ty, builder.getI64IntegerAttr(1));

          desc = builder.create<LLVM::InsertValueOp>(
              loc, desc, sentinel, ArrayRef<int64_t>{3, 0});
          desc = builder.create<LLVM::InsertValueOp>(
              loc, desc, one, ArrayRef<int64_t>{4, 0});
        }
        // Higher-rank descriptors can be extended here as needed.

        castOp.getResult(0).replaceAllUsesWith(desc);
        eraseCastChain(castOp);
        ++numIntToDescriptor;
        continue;
      }
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Plugin entry point
//===----------------------------------------------------------------------===//

extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "FlatClosureLowering", "0.1.0", []() {
            PassRegistration<ResolveClosureCastsPass>();
          }};
}
