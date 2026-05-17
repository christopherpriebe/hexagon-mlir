option(HEXAGON_MLIR_BUILD_RUNTIME
       "Build qcom_hexagon_backend/bin/runtime/ (needs Hexagon SDK + Tools + HexKL)."
       ON)
option(HEXAGON_MLIR_BUILD_TRANSLATE
       "Build the linalg-hexagon-translate executable (links hexagon_runtime)."
       ON)
option(HEXAGON_MLIR_BUILD_LSP_SERVER
       "Build the hexagon-mlir-lsp-server executable."
       ON)
option(HEXAGON_MLIR_OPT_LINK_EXECUTION_ENGINE
       "Link MLIRExecutionEngine into linalg-hexagon-opt. Required for JIT \
execution (e.g. by mlir-cpu-runner). The opt-only IR profiling path doesn't \
use it; turn off to skip the dependency, which lets LLVM be built without \
including the host arch in LLVM_TARGETS_TO_BUILD."
       ON)
option(HEXAGON_MLIR_BUILD_OPT
       "Build the linalg-hexagon-opt executable. The Triton plugin build \
doesn't need it (Triton compiles in-process via Python bindings), and \
attempting to link it in Triton's context pulls in TritonGPU/TritonNvidiaGPU \
IR libs that have unresolved references to internal Utility helpers \
(inferDstEncoding, verifyBarrierType). Set OFF when building via Triton's \
setup.py; ON in the standalone build_opt.sh wrapper."
       ON)

if(HEXAGON_MLIR_OPT_LINK_EXECUTION_ENGINE)
  set(_OPT_EXECUTION_ENGINE_LIB MLIRExecutionEngine)
else()
  set(_OPT_EXECUTION_ENGINE_LIB "")
endif()

if(HEXAGON_MLIR_BUILD_RUNTIME)
  add_subdirectory(runtime)
endif()

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

###############################################################################
# hexagon-translate
###############################################################################

if(HEXAGON_MLIR_BUILD_TRANSLATE)
add_llvm_executable(linalg-hexagon-translate
    linalg-hexagon-translate.cpp
    LinkRuntimeModules.cpp
    PARTIAL_SOURCES_INTENDED
)

llvm_update_compile_flags(linalg-hexagon-translate)
target_link_libraries(linalg-hexagon-translate PRIVATE
    HexagonMLIRLLVMToLLVMIR
    ${dialect_libs}
    ${conversion_libs}
    # tests
    TritonTestAnalysis

    LLVMCore
    LLVMSupport
    LLVMOption
    LLVMCodeGen
    LLVMAsmParser

    # MLIR core
    MLIROptLib
    MLIRIR
    MLIRLLVMDialect
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRExecutionEngine
    MLIRMathDialect
    MLIRMathToLLVM
    MLIRTransformUtils
    MLIRLLVMToLLVMIRTranslation

    # Hexagon runtime routines
    hexagon_runtime
)

mlir_check_all_link_libraries(linalg-hexagon-translate)
endif()

###############################################################################
# linalg-hexagon-opt
###############################################################################

if(HEXAGON_MLIR_BUILD_OPT)
add_llvm_executable(linalg-hexagon-opt linalg-hexagon-opt.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(linalg-hexagon-opt)
target_link_libraries(linalg-hexagon-opt PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    # tests
    TritonTestAnalysis

    LLVMCore
    LLVMSupport
    LLVMOption
    LLVMCodeGen
    LLVMAsmParser

    # MLIR core
    MLIROptLib
    MLIRIR
    MLIRLLVMDialect
    MLIRPass
    MLIRSupport
    MLIRTransforms
    ${_OPT_EXECUTION_ENGINE_LIB}
    MLIRMathToLLVM
    MLIRTransformUtils
    MLIRLLVMToLLVMIRTranslation
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRRegisterAllExtensions
)

mlir_check_all_link_libraries(linalg-hexagon-opt)
endif()

###############################################################################
# hexagon-mlir-lsp-server
###############################################################################

if(HEXAGON_MLIR_BUILD_LSP_SERVER)
add_llvm_executable(hexagon-mlir-lsp-server hexagon-mlir-lsp-server.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(hexagon-mlir-lsp-server)
target_link_libraries(hexagon-mlir-lsp-server PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    # tests
    TritonTestAnalysis

    # MLIR core
    MLIRIR
    MLIRLspServerLib
    MLIRRegisterAllExtensions
    MLIRRegisterAllDialects
)

mlir_check_all_link_libraries(hexagon-mlir-lsp-server)
endif()
