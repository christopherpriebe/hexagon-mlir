list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(LINALG_HEXAGON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(LINALG_HEXAGON_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/bin)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files
include_directories(${Python3_INCLUDE_DIR})
include_directories(${pybind11_INCLUDE_DIR})

option(HEXAGON_MLIR_BUILD_LIT_TESTS
       "Configure the FileCheck/lit testsuite under qcom_hexagon_backend/test/."
       ON)
option(HEXAGON_MLIR_LINK_RUNTIME_MODULES
       "Build the TritonHexagon plugin against the real LinkRuntimeModules \
implementation, which links nine Hexagon-runtime bitcode bundles into each \
compiled kernel and therefore depends on the hexagon_runtime CMake target \
(which in turn needs the Hexagon SDK / Tools / HexKL). Set OFF to compile \
a no-op stub instead — appropriate for IR-profiling builds that don't need \
device dispatch."
       ON)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(bin)
if(HEXAGON_MLIR_BUILD_LIT_TESTS)
  add_subdirectory(test)
endif()

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

if(HEXAGON_MLIR_LINK_RUNTIME_MODULES)
  set(_LINK_RUNTIME_SRC ${CMAKE_CURRENT_SOURCE_DIR}/bin/LinkRuntimeModules.cpp)
  set(_LINK_RUNTIME_LIB hexagon_runtime)
else()
  set(_LINK_RUNTIME_SRC ${CMAKE_CURRENT_SOURCE_DIR}/bin/LinkRuntimeModulesStub.cpp)
  set(_LINK_RUNTIME_LIB "")
  message(STATUS "qcom_hexagon_backend: TritonHexagon plugin will use the "
                 "no-op linkRuntimeModules stub. The plugin builds without "
                 "hexagon_runtime; compiled kernels CANNOT execute on device.")
endif()

add_triton_plugin(TritonHexagon
    ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_qcom_hexagon_backend_api.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_qcom_hexagon_backend.cc
    ${_LINK_RUNTIME_SRC}

    LINK_LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    TritonLinalgLLVM
    AffineToLLVM
    LinalgToLLVM
    DMAToLLVM
    HexKLToLLVM
    HexagonMemToLLVM
    HexagonMLIRLLVMToLLVMIR
    Transforms

    MLIRRegisterAllDialects

    # Hexagon runtime routines (skipped when HEXAGON_MLIR_LINK_RUNTIME_MODULES=OFF)
    ${_LINK_RUNTIME_LIB}
)
