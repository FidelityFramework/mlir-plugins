#!/usr/bin/env bash
# Run all flat-closure-lowering tests.
#
# Usage: ./run_tests.sh [path-to-plugin.so]
#
# If no plugin path is given, looks in ../build/flat-closure-lowering/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN="${1:-${SCRIPT_DIR}/../../build/flat-closure-lowering/flat-closure-lowering.so}"

if [ ! -f "$PLUGIN" ]; then
  echo "ERROR: Plugin not found at $PLUGIN"
  echo "Build first: cmake -B build -G Ninja && cmake --build build"
  exit 1
fi

PASS=0
FAIL=0

run_test() {
  local name="$1"
  local input="$2"
  local pipeline="$3"

  printf "  %-30s " "$name"

  # Run mlir-opt with plugin
  if ! mlir-opt \
    --load-pass-plugin="$PLUGIN" \
    --pass-pipeline="$pipeline" \
    "$input" \
    -o /tmp/mlir-test-output.mlir 2>/tmp/mlir-test-err.txt; then
    echo "FAIL (mlir-opt)"
    cat /tmp/mlir-test-err.txt
    FAIL=$((FAIL + 1))
    return
  fi

  # Translate to LLVM IR
  if ! mlir-translate --mlir-to-llvmir /tmp/mlir-test-output.mlir \
    -o /tmp/mlir-test-output.ll 2>/tmp/mlir-test-err.txt; then
    echo "FAIL (mlir-translate)"
    cat /tmp/mlir-test-err.txt
    FAIL=$((FAIL + 1))
    return
  fi

  # Verify no unrealized_conversion_cast remains
  if grep -q "unrealized_conversion_cast" /tmp/mlir-test-output.mlir; then
    echo "FAIL (unresolved casts remain)"
    grep "unrealized_conversion_cast" /tmp/mlir-test-output.mlir
    FAIL=$((FAIL + 1))
    return
  fi

  echo "PASS"
  PASS=$((PASS + 1))
}

FULL_PIPELINE="builtin.module(expand-strided-metadata,memref-expand,finalize-memref-to-llvm,convert-vector-to-llvm,convert-scf-to-cf,convert-cf-to-llvm,convert-index-to-llvm,convert-func-to-llvm,convert-arith-to-llvm,resolve-closure-casts,reconcile-unrealized-casts,canonicalize)"
SIMPLE_PIPELINE="builtin.module(convert-func-to-llvm,convert-arith-to-llvm,resolve-closure-casts,reconcile-unrealized-casts,canonicalize)"

echo "flat-closure-lowering tests"
echo "Plugin: $PLUGIN"
echo ""

run_test "closure_pair"    "$SCRIPT_DIR/closure_pair.mlir"    "$FULL_PIPELINE"
run_test "capture_memref"  "$SCRIPT_DIR/capture_memref.mlir"  "$FULL_PIPELINE"
run_test "multi_capture"   "$SCRIPT_DIR/multi_capture.mlir"   "$FULL_PIPELINE"
run_test "passthrough"     "$SCRIPT_DIR/passthrough.mlir"     "$SIMPLE_PIPELINE"

echo ""
echo "Results: $PASS passed, $FAIL failed"

if [ $FAIL -ne 0 ]; then
  exit 1
fi
