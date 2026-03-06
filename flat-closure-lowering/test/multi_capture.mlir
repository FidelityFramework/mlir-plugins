// RUN: mlir-opt %s \
// RUN:   --load-pass-plugin=%plugin \
// RUN:   --pass-pipeline="builtin.module(expand-strided-metadata,memref-expand,finalize-memref-to-llvm,convert-func-to-llvm,convert-index-to-llvm,convert-arith-to-llvm,resolve-closure-casts,reconcile-unrealized-casts,canonicalize)" \
// RUN:   -o %t.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.mlir | FileCheck %s

// Test: Multiple closures with different signatures.
//
// Exercises the pass with varied function signatures to ensure the ghost
// function type in the cast chain doesn't cause issues regardless of arity
// or return type.

// Zero-argument closure (thunk)
func.func @make_thunk() -> index {
  %fn = func.constant @thunk_impl : () -> index
  %code = builtin.unrealized_conversion_cast %fn : () -> index to index
  return %code : index
}

func.func @call_thunk(%code: index) -> index {
  %fn = builtin.unrealized_conversion_cast %code : index to () -> index
  %result = func.call_indirect %fn() : () -> index
  return %result : index
}

// Multi-argument closure
func.func @make_binary_op() -> index {
  %fn = func.constant @add_impl : (index, index) -> index
  %code = builtin.unrealized_conversion_cast %fn : (index, index) -> index to index
  return %code : index
}

func.func @call_binary_op(%code: index, %a: index, %b: index) -> index {
  %fn = builtin.unrealized_conversion_cast %code : index to (index, index) -> index
  %result = func.call_indirect %fn(%a, %b) : (index, index) -> index
  return %result : index
}

// Boolean-returning closure (comparison predicate)
func.func @make_predicate() -> index {
  %fn = func.constant @pred_impl : (index) -> i1
  %code = builtin.unrealized_conversion_cast %fn : (index) -> i1 to index
  return %code : index
}

func.func @call_predicate(%code: index, %arg: index) -> i1 {
  %fn = builtin.unrealized_conversion_cast %code : index to (index) -> i1
  %result = func.call_indirect %fn(%arg) : (index) -> i1
  return %result : i1
}

func.func private @thunk_impl() -> index
func.func private @add_impl(index, index) -> index
func.func private @pred_impl(index) -> i1

// CHECK-LABEL: define i64 @make_thunk
// CHECK:         ptrtoint ptr @thunk_impl to i64
//
// CHECK-LABEL: define i64 @call_thunk
// CHECK:         inttoptr i64 %{{.*}} to ptr
// CHECK:         call i64 %{{.*}}()
//
// CHECK-LABEL: define i64 @make_binary_op
// CHECK:         ptrtoint ptr @add_impl to i64
//
// CHECK-LABEL: define i64 @call_binary_op
// CHECK:         inttoptr i64 %{{.*}} to ptr
// CHECK:         call i64 %{{.*}}(i64 {{.*}}, i64
//
// CHECK-LABEL: define i64 @make_predicate
// CHECK:         ptrtoint ptr @pred_impl to i64
//
// CHECK-LABEL: define i1 @call_predicate
// CHECK:         inttoptr i64 %{{.*}} to ptr
// CHECK:         call i1 %{{.*}}(i64
