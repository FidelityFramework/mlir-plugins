# reconcile-ffi-externs — MLIR pass plugin for the Fidelity Framework
#
# Reconciles FFI extern declarations (ffi.* prefix) with MLIR infrastructure
# declarations after standard dialect lowering. Strips the ffi. prefix and
# inserts type casts (ptrtoint/inttoptr) where signatures differ.
#
# Loaded via: mlir-opt --load-pass-plugin=reconcile-ffi-externs.so \
#                      --reconcile-ffi-externs

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl)
add_public_tablegen_target(ReconcileFFIExternsPassIncGen)

add_llvm_pass_plugin(reconcile-ffi-externs
  ReconcileFFIExterns.cpp

  DEPENDS
  ReconcileFFIExternsPassIncGen
)

target_include_directories(reconcile-ffi-externs PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}  # For generated .h.inc files
)

target_link_libraries(reconcile-ffi-externs PRIVATE
  MLIRIR
  MLIRPass
  MLIRLLVMDialect
)
