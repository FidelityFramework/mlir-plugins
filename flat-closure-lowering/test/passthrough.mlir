// RUN: mlir-opt %s \
// RUN:   --load-pass-plugin=%plugin \
// RUN:   --pass-pipeline="builtin.module(convert-func-to-llvm,convert-arith-to-llvm,resolve-closure-casts,reconcile-unrealized-casts,canonicalize)" \
// RUN:   -o %t.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.mlir | FileCheck %s

// Test: Pass is transparent to code that doesn't use closures.
//
// Ensures the plugin doesn't interfere with non-closure code. This is the
// "transparent to any application that doesn't need it" property.

func.func @add(%a: index, %b: index) -> index {
  %result = arith.addi %a, %b : index
  return %result : index
}

func.func @constant() -> index {
  %c42 = arith.constant 42 : index
  return %c42 : index
}

// CHECK-LABEL: define i64 @add(i64 %0, i64 %1)
// CHECK:         add i64
//
// CHECK-LABEL: define i64 @constant
// CHECK:         ret i64 42
