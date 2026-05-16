option(HEXAGON_MLIR_BUILD_RUNTIME
       "Build qcom_hexagon_backend/bin/runtime/ (needs Hexagon SDK + Tools + HexKL)."
       ON)
option(HEXAGON_MLIR_BUILD_TRANSLATE
       "Build the linalg-hexagon-translate executable (links hexagon_runtime)."
       ON)
option(HEXAGON_MLIR_BUILD_LSP_SERVER
       "Build the hexagon-mlir-lsp-server executable."
       ON)

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
    MLIRExecutionEngine
    MLIRMathToLLVM
    MLIRTransformUtils
    MLIRLLVMToLLVMIRTranslation
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRRegisterAllExtensions
)

mlir_check_all_link_libraries(linalg-hexagon-opt)

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
