// Test: Simple rename — no collision with MLIR infrastructure.
//
// @ffi.wl_display_connect should be renamed to @wl_display_connect with no
// type changes. This is the common case for the vast majority of FFI externs
// (Wayland, DRM, GBM, resvg, etc.).

llvm.func @ffi.wl_display_connect(!llvm.ptr) -> !llvm.ptr attributes {sym_visibility = "private"}

llvm.func @caller(%name: !llvm.ptr) -> !llvm.ptr {
  %result = llvm.call @ffi.wl_display_connect(%name) : (!llvm.ptr) -> !llvm.ptr
  llvm.return %result : !llvm.ptr
}

// CHECK-LABEL: define {{.*}} @caller
// CHECK:         call ptr @wl_display_connect(ptr
// CHECK-NOT:     @ffi.wl_display_connect
