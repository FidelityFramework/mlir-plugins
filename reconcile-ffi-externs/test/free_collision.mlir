// Test: free collision — argument type reconciliation.
//
// @ffi.free(i64) collides with @free(!llvm.ptr). The pass should insert
// llvm.inttoptr before the call to convert the i64 argument to !llvm.ptr.

// Infrastructure declaration (from finalize-memref-to-llvm)
llvm.func @free(!llvm.ptr)

// FFI declaration (from Composer)
llvm.func @ffi.free(i64) attributes {sym_visibility = "private"}

llvm.func @deallocate(%ptr_as_int: i64) {
  llvm.call @ffi.free(%ptr_as_int) : (i64) -> ()
  llvm.return
}

// CHECK-LABEL: define {{.*}} @deallocate
// CHECK:         %[[PTR:.*]] = inttoptr i64 %{{.*}} to ptr
// CHECK:         call void @free(ptr %[[PTR]])
// CHECK-NOT:     @ffi.free
