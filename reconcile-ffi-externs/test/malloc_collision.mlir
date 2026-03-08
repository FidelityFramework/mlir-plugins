// Test: malloc collision — FFI extern vs MLIR infrastructure.
//
// @ffi.malloc(i64) -> i64 collides with @malloc(i64) -> !llvm.ptr
// (created by finalize-memref-to-llvm for memref.alloc lowering).
//
// The pass should:
// 1. Rewrite @ffi.malloc calls to use @malloc (infrastructure version)
// 2. Insert llvm.ptrtoint after the call (ptr → i64)
// 3. Remove the @ffi.malloc declaration
//
// Both the infrastructure @malloc (for memref.alloc) and the FFI @ffi.malloc
// (for user-facing Fidelity.Libc.Memory.malloc) resolve to the same libc
// symbol at link time.

// Infrastructure declaration (from finalize-memref-to-llvm)
llvm.func @malloc(i64) -> !llvm.ptr

// FFI declaration (from Composer's pExternCallResolved)
llvm.func @ffi.malloc(i64) -> i64 attributes {sym_visibility = "private"}

llvm.func @allocate_buffer(%size: i64) -> i64 {
  // FFI call: user code calling malloc through Fidelity.Libc.Memory
  %ptr_as_int = llvm.call @ffi.malloc(%size) : (i64) -> i64
  llvm.return %ptr_as_int : i64
}

llvm.func @infra_user(%size: i64) -> !llvm.ptr {
  // Infrastructure call: memref.alloc lowering
  %ptr = llvm.call @malloc(%size) : (i64) -> !llvm.ptr
  llvm.return %ptr : !llvm.ptr
}

// CHECK-LABEL: define {{.*}} @allocate_buffer
// CHECK:         %[[PTR:.*]] = call ptr @malloc(i64
// CHECK:         %[[INT:.*]] = ptrtoint ptr %[[PTR]] to i64
// CHECK:         ret i64 %[[INT]]
//
// CHECK-LABEL: define {{.*}} @infra_user
// CHECK:         %[[PTR:.*]] = call ptr @malloc(i64
// CHECK:         ret ptr %[[PTR]]
//
// CHECK-NOT:     @ffi.malloc
