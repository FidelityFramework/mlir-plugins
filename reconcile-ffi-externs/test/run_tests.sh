#!/usr/bin/env bash
# Run all reconcile-ffi-externs tests.
#
# Usage: ./run_tests.sh [path-to-plugin.so]
#
# If no plugin path is given, looks in ../build/reconcile-ffi-externs/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN="${1:-${SCRIPT_DIR}/../../build/reconcile-ffi-externs/reconcile-ffi-externs.so}"

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
    -o /tmp/mlir-ffi-test-output.mlir 2>/tmp/mlir-ffi-test-err.txt; then
    echo "FAIL (mlir-opt)"
    cat /tmp/mlir-ffi-test-err.txt
    FAIL=$((FAIL + 1))
    return
  fi

  # Translate to LLVM IR
  if ! mlir-translate --mlir-to-llvmir /tmp/mlir-ffi-test-output.mlir \
    -o /tmp/mlir-ffi-test-output.ll 2>/tmp/mlir-ffi-test-err.txt; then
    echo "FAIL (mlir-translate)"
    cat /tmp/mlir-ffi-test-err.txt
    FAIL=$((FAIL + 1))
    return
  fi

  # Verify no ffi.* symbols remain in LLVM IR
  if grep -q '@ffi\.' /tmp/mlir-ffi-test-output.ll; then
    echo "FAIL (ffi.* symbols remain in LLVM IR)"
    grep '@ffi\.' /tmp/mlir-ffi-test-output.ll
    FAIL=$((FAIL + 1))
    return
  fi

  echo "PASS"
  PASS=$((PASS + 1))
}

# Pipeline: reconcile-ffi-externs runs in LLVM dialect, then canonicalize
PIPELINE="builtin.module(reconcile-ffi-externs,canonicalize)"

echo "reconcile-ffi-externs tests"
echo "Plugin: $PLUGIN"
echo ""

run_test "simple_rename"     "$SCRIPT_DIR/simple_rename.mlir"     "$PIPELINE"
run_test "malloc_collision"  "$SCRIPT_DIR/malloc_collision.mlir"   "$PIPELINE"
run_test "free_collision"    "$SCRIPT_DIR/free_collision.mlir"     "$PIPELINE"

echo ""
echo "Results: $PASS passed, $FAIL failed"

if [ $FAIL -ne 0 ]; then
  exit 1
fi
