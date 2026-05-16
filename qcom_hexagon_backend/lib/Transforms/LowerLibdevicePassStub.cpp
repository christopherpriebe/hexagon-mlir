//===- LowerLibdevicePassStub.cpp -----------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// No-op replacement for LowerLibdevicePass.cpp compiled when
// HEXAGON_MLIR_BUILD_LIBDEVICE_LOWERING=OFF. The real pass depends on
// Triton's TritonGPU dialect headers (and therefore on a Triton source
// tree); downstream consumers that build qcom_hexagon_backend stand-
// alone — e.g. for IR profiling without the full Triton chain — do not
// have those headers available.
//
// The stub still satisfies the `createLowerLibdevicePass()` symbol so
// the LinalgToLLVM mega-pass at line 202 of LinalgToLLVMPass.cpp can
// `pm.addPass(createLowerLibdevicePass())` without losing its slot in
// the pass order. When the input MLIR has no `tt.extern_elementwise`
// ops (the only thing the real pass acts on), the no-op stub is
// functionally equivalent — exactly the case for the pre-lowered MLIR
// benchmarks under test/python/mlir/.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

namespace mlir::hexagon {

namespace {
struct LowerLibdeviceStubPass
    : public PassWrapper<LowerLibdeviceStubPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "lower-libdevice"; }
  StringRef getDescription() const final {
    return "no-op stub (libdevice lowering disabled at build time via "
           "HEXAGON_MLIR_BUILD_LIBDEVICE_LOWERING=OFF)";
  }
  void runOnOperation() final { /* no-op */ }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createLowerLibdevicePass() {
  return std::make_unique<LowerLibdeviceStubPass>();
}

} // namespace mlir::hexagon
