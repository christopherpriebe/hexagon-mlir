//===- LinkRuntimeModulesStub.cpp -----------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// No-op replacement for LinkRuntimeModules.cpp compiled when
// HEXAGON_MLIR_LINK_RUNTIME_MODULES=OFF. The real implementation links
// nine Hexagon-runtime bitcode bundles (IntrinsicsHVX, VTCMPool, HexagonAPI,
// HexagonBuffer, HexagonBufferAlias, HexagonCAPI, HexKLAPI, RuntimeDMA,
// UserDMA) into the compiled kernel. Those bundles are produced by the
// `hexagon_runtime` CMake target, which in turn requires the Hexagon SDK
// + Hexagon Tools + HexKL — Qualcomm-account-gated downloads.
//
// For IR-profiling builds we don't need device dispatch, only the MLIR
// pass output. The pass-manager `enableIRPrinting` hook in
// MLLVMIRTranslation.cpp fires DURING the pass run, well before this
// post-translation linking step. So a no-op stub here lets the TritonHexagon
// plugin build standalone (no Hexagon SDK) while still capturing every
// per-pass IR snapshot.
//
// The resulting kernel module is missing runtime symbol definitions and
// CANNOT execute on device. That's intentional and harmless for the
// splice-study workflow.
//
//===----------------------------------------------------------------------===//

#include "LinkRuntimeModules.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace mlir::Hexagon::Translate {

void linkRuntimeModules(
    llvm::LLVMContext &ctx, std::unique_ptr<llvm::Module> &module,
    const std::unordered_map<std::string, std::string> &arch_kwargs) {
  // Intentionally empty. Marker compiles silently — no llvm::errs output
  // because this gets called per kernel and the per-call noise would
  // bury the actual IR dumps in pytest stderr.
  (void)ctx;
  (void)module;
  (void)arch_kwargs;
}

} // namespace mlir::Hexagon::Translate
