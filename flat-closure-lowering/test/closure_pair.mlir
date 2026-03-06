// RUN: mlir-opt %s \
// RUN:   --load-pass-plugin=%plugin \
// RUN:   --pass-pipeline="builtin.module(expand-strided-metadata,memref-expand,finalize-memref-to-llvm,convert-func-to-llvm,convert-index-to-llvm,convert-arith-to-llvm,resolve-closure-casts,reconcile-unrealized-casts,canonicalize)" \
// RUN:   -o %t.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.mlir | FileCheck %s

// Test: Flat closure construction and invocation.
//
// The MiddleEnd stores a function address as index (via FuncToIndex) in a
// memref<2xindex> pair. On call, it loads the index and recovers the function
// type (via IndexToFunc) for call_indirect.
//
// Expected lowering:
//   FuncToIndex → llvm.ptrtoint
//   IndexToFunc → llvm.inttoptr + indirect call

func.func @make_closure() -> memref<2xindex> {
  %fn = func.constant @impl : (index) -> index
  %code = builtin.unrealized_conversion_cast %fn : (index) -> index to index

  %pair = memref.alloca() : memref<2xindex>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %env = arith.constant 99 : index
  memref.store %code, %pair[%c0] : memref<2xindex>
  memref.store %env, %pair[%c1] : memref<2xindex>
  return %pair : memref<2xindex>
}

func.func @call_closure(%pair: memref<2xindex>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %code_idx = memref.load %pair[%c0] : memref<2xindex>
  %env_idx = memref.load %pair[%c1] : memref<2xindex>
  %fn = builtin.unrealized_conversion_cast %code_idx : index to (index) -> index
  %result = func.call_indirect %fn(%env_idx) : (index) -> index
  return %result : index
}

func.func private @impl(index) -> index

// CHECK-LABEL: define {{.*}} @make_closure
// CHECK:         ptrtoint ptr @impl to i64
// CHECK:         store i64
//
// CHECK-LABEL: define {{.*}} @call_closure
// CHECK:         %[[CODE:.*]] = load i64, ptr
// CHECK:         %[[PTR:.*]] = inttoptr i64 %[[CODE]] to ptr
// CHECK:         call i64 %[[PTR]](i64
