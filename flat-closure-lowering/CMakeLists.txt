# flat-closure-lowering — MLIR pass plugin for the Fidelity Framework
#
# Resolves unrealized_conversion_cast operations that arise from flat closure
# representation in the MiddleEnd (Alex). These casts bridge the semantic gap
# between target-agnostic closure IR and LLVM's pointer/integer model.
#
# Loaded via: mlir-opt --load-pass-plugin=flat-closure-lowering.so \
#                      --resolve-closure-casts

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl)
add_public_tablegen_target(FlatClosureLoweringPassIncGen)

add_llvm_pass_plugin(flat-closure-lowering
  FlatClosureLowering.cpp

  DEPENDS
  FlatClosureLoweringPassIncGen
)

target_include_directories(flat-closure-lowering PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}  # For generated .h.inc files
)

target_link_libraries(flat-closure-lowering PRIVATE
  MLIRIR
  MLIRPass
  MLIRLLVMDialect
)
