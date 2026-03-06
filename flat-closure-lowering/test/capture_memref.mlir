// RUN: mlir-opt %s \
// RUN:   --load-pass-plugin=%plugin \
// RUN:   --pass-pipeline="builtin.module(expand-strided-metadata,memref-expand,finalize-memref-to-llvm,convert-func-to-llvm,convert-index-to-llvm,convert-arith-to-llvm,resolve-closure-casts,reconcile-unrealized-casts,canonicalize)" \
// RUN:   -o %t.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.mlir | FileCheck %s

// Test: Captured memref reconstruction from raw pointer (index).
//
// The MiddleEnd stores a memref pointer as index in the closure environment.
// On extraction, it casts back to memref<?xi8> (IndexToMemRef) and uses
// reinterpret_cast to establish correct bounds before access.
//
// Expected lowering:
//   IndexToMemRef → llvm.inttoptr + memref descriptor construction
//   (canonicalization may collapse descriptor + reinterpret_cast to direct load)

func.func @extract_capture(%env_ptr: index, %size: index) -> i8 {
  %env = builtin.unrealized_conversion_cast %env_ptr : index to memref<?xi8>
  %bounded = memref.reinterpret_cast %env to
    offset: [0], sizes: [%size], strides: [1]
    : memref<?xi8> to memref<?xi8>
  %c0 = arith.constant 0 : index
  %val = memref.load %bounded[%c0] : memref<?xi8>
  return %val : i8
}

// CHECK-LABEL: define i8 @extract_capture(i64 %0, i64 %1)
// CHECK:         inttoptr i64 %0 to ptr
// CHECK:         load i8, ptr
