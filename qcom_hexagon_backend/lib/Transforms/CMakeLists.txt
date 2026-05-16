option(HEXAGON_MLIR_BUILD_LIBDEVICE_LOWERING
       "Build the real LowerLibdevice pass. Requires Triton's TritonGPU \
dialect headers (which are only present when Triton is built as part of \
the qcom_hexagon_backend plugin). Set OFF to use a no-op stub instead — \
appropriate when running pre-lowered MLIR that has no tt.extern_elementwise \
ops, e.g. for IR profiling without the Triton build."
       ON)

if(HEXAGON_MLIR_BUILD_LIBDEVICE_LOWERING)
  set(_LOWER_LIBDEVICE_SRC LowerLibdevicePass.cpp)
else()
  set(_LOWER_LIBDEVICE_SRC LowerLibdevicePassStub.cpp)
  message(STATUS "qcom_hexagon_backend: LowerLibdevice replaced by no-op stub "
                 "(HEXAGON_MLIR_BUILD_LIBDEVICE_LOWERING=OFF).")
endif()

add_mlir_library(Transforms
  DoubleBufferGenericS1.cpp
  DoubleBufferGenericS2.cpp
  Transforms.cpp
  OptionsParsing.cpp
  ${_LOWER_LIBDEVICE_SRC}
  AddFastMathPass.cpp
  UseHexagonRoutines.cpp
  PuntBufferPass.cpp
  RVO.cpp
  ScheduleMatmulForHVXPass.cpp
  GeneralizePass.cpp
  ReplaceWithLibraryCallsPass.cpp
  VectorLoweringPass.cpp
  DecomposeTensorConcatPass.cpp
  MatmulToConvPass.cpp
  MatmulToHexKLPass.cpp
  CopyCanonicalizationPass.cpp
  DecomposeHexKLMatmulPass.cpp
  ConvTilingPass.cpp
  ConvertLayoutPass.cpp
  ExpandMathOpsPass.cpp
  RemoveMLProgram.cpp
  ExpandBoolVec.cpp
  EraseUnusedLinalgOperands.cpp
  CollapseAddressSpace.cpp
  FastInversePass.cpp
  ConvertZeroSizeMemrefPass.cpp
  ConvertToHexagonmem.cpp
  LowerTPtr.cpp
  SeedLayoutConversions.cpp
  ForceHVXCrouton.cpp
  SmallExponentToMultiplyPass.cpp
  LinalgUtils.cpp
  PackUnpackUtils.cpp
  PreprocessTiledConv2D.cpp
  InsertScratchArgPass.cpp
  MemoryOffsetsPass.cpp
  HexagonLWPInstrumentation.cpp
  LowerTmTensor.cpp
  LowerTTX.cpp
  ReduceContractionRankPass.cpp
  FoldCastsIntoMatmul.cpp
  HoistScalarOps.cpp
  FoldMulFByZeroPass.cpp
  FoldResourceTransposePass.cpp
  LowerHexKLMatmulToMacroPass.cpp
  PreprocessWeightsForHMXPass.cpp
  FoldPackUnpackConstantsPass.cpp
  EliminateRedundantUnpackPackPass.cpp
  DivToMulOptimizationPass.cpp
  SCFLoopUnrollPass.cpp

  DEPENDS
  TransformsPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRMathTransforms
  MLIRSupport
)
