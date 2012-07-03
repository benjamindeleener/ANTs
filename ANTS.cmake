
include(${CMAKE_CURRENT_LIST_DIR}/Common.cmake)

configure_file(${CMAKE_CURRENT_LIST_DIR}/CTestCustom.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/CTestCustom.cmake COPYONLY)

set(BUILDSCRIPTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/BuildScripts)
set(CMAKE_MODULE_PATH
    ${BUILDSCRIPTS_DIR}
    ${${PROJECT_NAME}_SOURCE_DIR}/CMake
    ${${PROJECT_NAME}_BINARY_DIR}/CMake
    ${CMAKE_MODULE_PATH}
    )

set (CMAKE_INCLUDE_DIRECTORIES_BEFORE ON)

# Set up ITK
find_package(ITK 4 REQUIRED)
include(${ITK_USE_FILE})

# Set up VTK
option(USE_VTK "Use VTK Libraries" OFF)
if(USE_VTK)
  find_package(VTK)
  if(VTK_FOUND)
    include(${VTK_USE_FILE})
  else(VTK_FOUND)
     message("Cannot build some programs without VTK.  Please set VTK_DIR if you need these programs.")
  endif(VTK_FOUND)
endif(USE_VTK)

# With MS compilers on Win64, we need the /bigobj switch, else generated
# code results in objects with number of sections exceeding object file
# format.
# see http://msdn.microsoft.com/en-us/library/ms173499.aspx
if(CMAKE_CL_64 OR MSVC)
  add_definitions(/bigobj)
endif()

option(USE_FFTWD "Use double precision fftw if found" OFF)
option(USE_FFTWF "Use single precision fftw if found" OFF)
option(USE_SYSTEM_FFTW "Use an installed version of fftw" OFF)
if (USE_FFTWD OR USE_FFTWF)
  if(USE_SYSTEM_FFTW)
      find_package( FFTW )
      link_directories(${FFTW_LIBDIR})
  else(USE_SYSTEM_FFTW)
      link_directories(${ITK_DIR}/fftw/lib)
      include_directories(${ITK_DIR}/fftw/include)
  endif(USE_SYSTEM_FFTW)
endif(USE_FFTWD OR USE_FFTWF)

#-----------------------------------------------------------------------------
include(CTest)
enable_testing()

# set(CMAKE_BUILD_TYPE "Release")
set(ANTS_TEST_BIN_DIR ${CMAKE_BINARY_DIR}/Examples)

add_subdirectory(Examples)

configure_file(${CMAKE_CURRENT_LIST_DIR}/CTestCustom.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/CTestCustom.cmake COPYONLY)

###
#  Perform testing
###
set(DATA_DIR ${CMAKE_SOURCE_DIR}/Examples/Data)
set(R16_IMAGE ${DATA_DIR}/r16slice.nii.gz)
set(R64_IMAGE ${DATA_DIR}/r64slice.nii.gz)
set(OUTPUT_PREFIX ${CMAKE_BINARY_DIR}/TEST)
set(WARP ${OUTPUT_PREFIX}Warp.nii.gz ${OUTPUT_PREFIX}Affine.txt )
set(INVERSEWARP -i ${OUTPUT_PREFIX}Affine.txt ${OUTPUT_PREFIX}InverseWarp.nii.gz )
set(WARP_IMAGE ${CMAKE_BINARY_DIR}/warped.nii.gz)
set(INVERSEWARP_IMAGE ${CMAKE_BINARY_DIR}/inversewarped.nii.gz)
set(DEVIL_IMAGE ${DATA_DIR}/Frown.nii)
set(ANGEL_IMAGE ${DATA_DIR}/Smile.nii)
set(SEG_IMAGE ${DATA_DIR}/nslice.nii.gz)
set(TEST_BINARY_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

###
#  ANTS metric testing
###

option(OLD_BASELINE_TESTS  "Use reported metrics from tests" OFF )

if(NOT OLD_BASELINE_TESTS)
  message(STATUS "New Baselines Values")
  # CC1 Testing
  add_test(ANTS_CC_1 ${TEST_BINARY_DIR}/ANTS 2 -m
    CC[ ${R16_IMAGE}, ${R64_IMAGE},  1 ,2  ] -r Gauss[ 3 , 0 ] -t SyN[ 0.5   ]
    -i 50x50x30 -o  ${OUTPUT_PREFIX}.nii.gz --number-of-affine-iterations 100x100x50 )

  add_test(ANTS_CC_1_WARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2
    ${R64_IMAGE} ${WARP_IMAGE} ${WARP} -R ${R16_IMAGE} )

  add_test(ANTS_CC_1_JPG  ${TEST_BINARY_DIR}/ConvertToJpg ${WARP_IMAGE} ANTSCC1.jpg)

  add_test(ANTS_CC_1_WARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity
    2 0 ${R16_IMAGE} ${WARP_IMAGE}
    ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 14.2984 0.05)

  add_test(ANTS_CC_1_WARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz 0.996429 0.05)

  add_test(ANTS_CC_1_WARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz -0.00038593 0.05)

  add_test(ANTS_CC_1_INVERSEWARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2
    ${R16_IMAGE} ${INVERSEWARP_IMAGE} ${INVERSEWARP}  -R ${R16_IMAGE}
    )

  add_test(ANTS_CC_1_INVERSEWARP_METRIC_0
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 50.4483 0.05)

  add_test(ANTS_CC_1_INVERSEWARP_METRIC_1
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 1.0 0.05)

  add_test(ANTS_CC_1_INVERSEWARP_METRIC_2
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 0.379897 0.05)

  add_test(ANTS_CC_2   ${TEST_BINARY_DIR}/ANTS 2 -m
    PR[${R16_IMAGE},${R64_IMAGE},1,4] -r Gauss[3,0] -t SyN[0.5] -i 50x50x30 -o
    ${OUTPUT_PREFIX}.nii.gz --go-faster true )
  add_test(ANTS_CC_2_WARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2
    ${R64_IMAGE} ${WARP_IMAGE} ${WARP}  -R ${R16_IMAGE} )
  add_test(ANTS_CC_2_JPG
    ${TEST_BINARY_DIR}/ConvertToJpg ${WARP_IMAGE} ANTSCC2.jpg)

  add_test(ANTS_CC_2_WARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz 14.3283 0.05)

  add_test(ANTS_CC_2_WARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz 0.996771 0.05)

  add_test(ANTS_CC_2_WARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz -0.000461792 0.05)

  add_test(ANTS_CC_2_INVERSEWARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2
    ${R16_IMAGE} ${INVERSEWARP_IMAGE} ${INVERSEWARP}   -R ${R16_IMAGE}
    )

  add_test(ANTS_CC_2_INVERSEWARP_METRIC_0
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
    50.4483 0.05)
  add_test(ANTS_CC_2_INVERSEWARP_METRIC_1
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
    1.0 0.05)
  add_test(ANTS_CC_2_INVERSEWARP_METRIC_2
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
    0.379897 0.05)

  #add_test(ANTS_CC_2_INVERSEWARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -5 0.05)
  add_test(ANTS_CC_3   ${TEST_BINARY_DIR}/ANTS 2 -m
    CC[${R16_IMAGE},${R64_IMAGE},1,4] -r Gauss[3,0] -t SyN[0.5] -i 50x50x30 -o
    ${OUTPUT_PREFIX}.nii.gz  --go-faster true )

  add_test(ANTS_CC_3_WARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2
    ${R64_IMAGE} ${WARP_IMAGE} ${WARP}  -R ${R16_IMAGE} )

  add_test(ANTS_CC_3_JPG
    ${TEST_BINARY_DIR}/ConvertToJpg ${WARP_IMAGE} ANTSCC2.jpg)

  add_test(ANTS_CC_3_WARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz 14.3283 0.05)

  add_test(ANTS_CC_3_WARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz 0.996796 0.05)

  add_test(ANTS_CC_3_WARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz -1.2204 0.05)

  add_test(ANTS_CC_3_INVERSEWARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2
    ${R16_IMAGE} ${INVERSEWARP_IMAGE} ${INVERSEWARP}   -R ${R16_IMAGE} )

  add_test(ANTS_CC_3_INVERSEWARP_METRIC_0
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
    50.4483 0.05)

  add_test(ANTS_CC_3_INVERSEWARP_METRIC_1
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R64_IMAGE} ${INVERSEWARP_IMAGE}
    ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 1.0 0.05)

  add_test(ANTS_CC_3_INVERSEWARP_METRIC_2
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
    0.379897 0.05)
  #add_test(ANTS_CC_3_INVERSEWARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -5 0.05)

  add_test(ANTS_MSQ    ${TEST_BINARY_DIR}/ANTS 2 -m
    MSQ[${R16_IMAGE},${R64_IMAGE},1,0] -r Gauss[3,0] -t SyN[0.25] -i 50x50x30 -o
    ${OUTPUT_PREFIX}.nii.gz)

  add_test(ANTS_MSQ_WARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2
    ${R64_IMAGE} ${WARP_IMAGE} ${WARP}  -R ${R16_IMAGE} )

  add_test(ANTS_MSQ_JPG  ${TEST_BINARY_DIR}/ConvertToJpg ${WARP_IMAGE} ANTSMSQ.jpg)

  add_test(ANTS_MSQ_WARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz 14.3426 0.05)

  add_test(ANTS_MSQ_WARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz 0.997077 0.05)

  add_test(ANTS_MSQ_WARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz -1.18658 0.05)

  add_test(ANTS_MSQ_INVERSEWARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2
    ${R16_IMAGE} ${INVERSEWARP_IMAGE} ${INVERSEWARP}   -R ${R16_IMAGE} )

  add_test(ANTS_MSQ_INVERSEWARP_METRIC_0
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
    50.4483 0.05)

  add_test(ANTS_MSQ_INVERSEWARP_METRIC_1
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
    1.0 0.05)

  add_test(ANTS_MSQ_INVERSEWARP_METRIC_2
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
    0.379897 0.05)

  add_test(ANTS_MI_1   ${TEST_BINARY_DIR}/ANTS 2 -m
    MI[${R16_IMAGE},${R64_IMAGE},1,32] -r Gauss[3,0] -t SyN[0.25] -i 50x50x30 -o
    ${OUTPUT_PREFIX}.nii.gz)

  add_test(ANTS_MI_1_WARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2
    ${R64_IMAGE} ${WARP_IMAGE} ${WARP}  -R ${R16_IMAGE} )

  add_test(ANTS_MI_1_JPG  ${TEST_BINARY_DIR}/ConvertToJpg ${WARP_IMAGE} ANTSMI1.jpg)

  add_test(ANTS_MI_1_WARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz 14.625 0.05)

  add_test(ANTS_MI_1_WARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz 0.992663 0.05)

  add_test(ANTS_MI_1_WARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz -1.04695 0.05)

  add_test(ANTS_MI_1_INVERSEWARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2
    ${R16_IMAGE} ${INVERSEWARP_IMAGE} ${INVERSEWARP}  -R ${R16_IMAGE} )

  add_test(ANTS_MI_1_INVERSEWARP_METRIC_0
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
    50.4483 0.05)

  add_test(ANTS_MI_1_INVERSEWARP_METRIC_1
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
    1.0 0.05)

  add_test(ANTS_MI_1_INVERSEWARP_METRIC_2
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
    0.379897 0.05)

  add_test(ANTS_MI_2   ${TEST_BINARY_DIR}/ANTS 2 -m
    MI[${R16_IMAGE},${R64_IMAGE},1,24] -r Gauss[3,0] -t SyN[0.25] -i 50x50x20 -o
    ${OUTPUT_PREFIX}.nii.gz)

  add_test(ANTS_MI_2_WARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2
    ${R64_IMAGE} ${WARP_IMAGE} ${WARP}  -R ${R16_IMAGE} )

  add_test(ANTS_MI_2_JPG  ${TEST_BINARY_DIR}/ConvertToJpg ${WARP_IMAGE} ANTSMI2.jpg)

  add_test(ANTS_MI_2_WARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz 14.545 0.05)

  add_test(ANTS_MI_2_WARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz 0.9936 0.05)

  add_test(ANTS_MI_2_WARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz 1.07288 0.05)

  add_test(ANTS_MI_2_INVERSEWARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2
    ${R16_IMAGE} ${INVERSEWARP_IMAGE} ${INVERSEWARP}  -R ${R16_IMAGE}  )

  add_test(ANTS_MI_2_INVERSEWARP_METRIC_0
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
    50.4483 0.05)

  add_test(ANTS_MI_2_INVERSEWARP_METRIC_1
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
    1.0 0.05)

  add_test(ANTS_MI_2_INVERSEWARP_METRIC_2
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
    0.379897 0.05)
  ###
  #  ANTS transform testing
  ###

  add_test(ANTS_ELASTIC ${TEST_BINARY_DIR}/ANTS 2 -m
    PR[${R16_IMAGE},${R64_IMAGE},1,2] -t Elast[1] -i 50x50x50 -r Gauss[0,1]
    -o ${OUTPUT_PREFIX}.nii.gz)

  add_test(ANTS_ELASTIC_WARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2
    ${R64_IMAGE} ${WARP_IMAGE} ${WARP}   -R ${R16_IMAGE}  )

  add_test(ANTS_ELASTIC_JPG  ${TEST_BINARY_DIR}/ConvertToJpg ${WARP_IMAGE}
    ANTSELASTIC.jpg)

  add_test(ANTS_ELASTIC_WARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2
    0 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz 14.6829 0.05)

  add_test(ANTS_ELASTIC_WARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2
    1 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz 0.989737 0.05)

  add_test(ANTS_ELASTIC_WARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2
    2 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz -1.01828 0.05)

  add_test(ANTS_GSYN    ${TEST_BINARY_DIR}/ANTS 2 -m
    CC[${R16_IMAGE},${R64_IMAGE},1,3] -t SyN[0.25] -i 50x50x50 -r Gauss[3,0.] -o
    ${OUTPUT_PREFIX}.nii.gz)

  add_test(ANTS_GSYN_WARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2
    ${R64_IMAGE} ${WARP_IMAGE} ${WARP}  -R ${R16_IMAGE} )

  add_test(ANTS_GSYN_JPG  ${TEST_BINARY_DIR}/ConvertToJpg ${WARP_IMAGE}
    ANTSGSYN.jpg)

  add_test(ANTS_GSYN_WARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz 14.2847 0.05)

  add_test(ANTS_GSYN_WARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz 9.996276 0.05)

  add_test(ANTS_GSYN_WARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz -1.19298 0.05)

  add_test(ANTS_GSYN_INVERSEWARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2
    ${R16_IMAGE} ${INVERSEWARP_IMAGE} ${INVERSEWARP}  -R ${R16_IMAGE}  )

  add_test(ANTS_GSYN_JPGINV  ${TEST_BINARY_DIR}/ConvertToJpg ${INVERSEWARP_IMAGE}
    ANTSGSYNINV.jpg)

  add_test(ANTS_GSYN_INVERSEWARP_METRIC_0
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
    50.4483 0.05)

  add_test(ANTS_GSYN_INVERSEWARP_METRIC_1
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
    1.0 0.05)

  add_test(ANTS_GSYN_INVERSEWARP_METRIC_2
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
    0.379897 0.05)

  add_test(ANTS_EXP ${TEST_BINARY_DIR}/ANTS 2 -m
    PR[${R16_IMAGE},${R64_IMAGE},1,4] -t Exp[0.5,2,0.5] -i 50x50x50 -r
    Gauss[0.5,0.25] -o ${OUTPUT_PREFIX}.nii.gz)

  add_test(ANTS_EXP_WARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2
    ${R64_IMAGE} ${WARP_IMAGE} ${WARP}  -R ${R16_IMAGE} )

  add_test(ANTS_EXP_JPG  ${TEST_BINARY_DIR}/ConvertToJpg ${WARP_IMAGE} ANTSEXP.jpg)

  add_test(ANTS_EXP_WARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz 14.3283 0.05)

  add_test(ANTS_EXP_WARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz 0.996139 0.05)

  add_test(ANTS_EXP_WARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz -1.14716 0.05)

  add_test(ANTS_EXP_INVERSEWARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2
    ${R16_IMAGE} ${INVERSEWARP_IMAGE} ${INVERSEWARP}   -R ${R16_IMAGE} )

  add_test(ANTS_EXP_INVERSEWARP_METRIC_0
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
    50.4483 0.05)

  add_test(ANTS_EXP_INVERSEWARP_METRIC_1
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
    1.0 0.05)

  add_test(ANTS_EXP_INVERSEWARP_METRIC_2
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
    0.379897 0.05)

  #add_test(ANTS_GSYN    ${TEST_BINARY_DIR}/ANTS 2 -m PR[${R16_IMAGE},${R64_IMAGE},1,2] -t SyN[0.75]            -i 50x50x50 -r Gauss[3,0.0,32] -o ${OUTPUT_PREFIX}.nii.gz)

  add_test(ANTS_SYN ${TEST_BINARY_DIR}/ANTS 2 -m
    PR[${R16_IMAGE},${R64_IMAGE},1,2] -t SyN[0.5,2,0.05] -i 50x50x50 -r
    Gauss[3,0.0,32] -o ${OUTPUT_PREFIX}.nii.gz)

  add_test(ANTS_SYN_WARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2
    ${R64_IMAGE} ${WARP_IMAGE} ${WARP}  -R ${R16_IMAGE}  )

  add_test(ANTS_SYN_JPG  ${TEST_BINARY_DIR}/ConvertToJpg ${WARP_IMAGE}
    ANTSSYN.jpg)

  add_test(ANTS_SYN_WARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz 14.3283 0.05)

  add_test(ANTS_SYN_WARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz 0.99559 0.05)

  add_test(ANTS_SYN_WARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2
    ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt
    ${OUTPUT_PREFIX}metric.nii.gz -1.18658 0.05)

  add_test(ANTS_SYN_INVERSEWARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2
    ${R16_IMAGE} ${INVERSEWARP_IMAGE} ${INVERSEWARP}  -R ${R16_IMAGE}  )

  add_test(ANTS_SYN_JPGINV  ${TEST_BINARY_DIR}/ConvertToJpg ${INVERSEWARP_IMAGE}
    ANTSSYNINV.jpg)

  add_test(ANTS_SYN_INVERSEWARP_METRIC_0
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
    50.4483 0.05)

  add_test(ANTS_SYN_INVERSEWARP_METRIC_1
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
    1.0 0.05)

  add_test(ANTS_SYN_INVERSEWARP_METRIC_2
    ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R64_IMAGE}
    ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
    0.37897 0.05)
  ###
  # PSE sub-tests:  Check to see if .txt files and .vtk files also run correctly
  ###
  set(ANGEL_IMAGE_TXT ${DATA_DIR}/Smile.txt)
  set(DEVIL_IMAGE_TXT ${DATA_DIR}/Frown.txt)
  set(ANGEL_IMAGE_VTK ${DATA_DIR}/Smile.vtk)
  set(DEVIL_IMAGE_VTK ${DATA_DIR}/Frown.vtk)

  add_test(ANTS_PSE_MSQ_TXT ${TEST_BINARY_DIR}/ANTS 2 -i 191x170x155x40x30  -r
    Gauss[3,0] -t SyN[0.2] -m MSQ[${DEVIL_IMAGE},${ANGEL_IMAGE},1,0] -m
    PSE[${DEVIL_IMAGE},${ANGEL_IMAGE},${DEVIL_IMAGE_TXT},${ANGEL_IMAGE_TXT},1,0.1,11,1,1]
    --continue-affine 0 --number-of-affine-iterations 0
    -o ${OUTPUT_PREFIX}.nii.gz   --use-all-metrics-for-convergence 1 )

  add_test(ANTS_PSE_MSQ_VTK ${TEST_BINARY_DIR}/ANTS 2 -i 91x70x55x40x30  -r
    Gauss[3,0.,32] -t SyN[0.25]
    -m MSQ[${DEVIL_IMAGE},${ANGEL_IMAGE},1,0]
    -m PSE[${DEVIL_IMAGE},${ANGEL_IMAGE},${DEVIL_IMAGE_VTK},${ANGEL_IMAGE_VTK},1,0.33,11,1,25]
    --continue-affine 0 --number-of-affine-iterations 0
    -o ${OUTPUT_PREFIX}.nii.gz)
else(NOT OLD_BASELINE_TESTS)
  message(STATUS "Old Baselines Values")
  ###
  #  ANTS metric testing
  ###
  add_test(ANTS_CC_1 ${TEST_BINARY_DIR}/ANTS 2 -m  CC[ ${R16_IMAGE}, ${R64_IMAGE},  1 ,2  ] -r Gauss[ 3 , 0 ] -t SyN[ 0.5   ] -i 50x50x30 -o  ${OUTPUT_PREFIX}.nii.gz --number-of-affine-iterations 100x100x50 )
  add_test(ANTS_CC_1_WARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2 ${R64_IMAGE} ${WARP_IMAGE} ${WARP} -R ${R16_IMAGE} )
  add_test(ANTS_CC_1_JPG  ${TEST_BINARY_DIR}/ConvertToJpg ${WARP_IMAGE} ANTSCC1.jpg)
  add_test(ANTS_CC_1_WARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 11.9992 0.05)
  add_test(ANTS_CC_1_WARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.61 0.05)
  add_test(ANTS_CC_1_WARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.00038593 0.05)
  add_test(ANTS_CC_1_INVERSEWARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2 ${R16_IMAGE} ${INVERSEWARP_IMAGE} ${INVERSEWARP}  -R ${R16_IMAGE} )
  add_test(ANTS_CC_1_INVERSEWARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 12.1606 0.05)
  add_test(ANTS_CC_1_INVERSEWARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.61 0.05)
  add_test(ANTS_CC_1_INVERSEWARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.000380545 0.05)
  add_test(ANTS_CC_2   ${TEST_BINARY_DIR}/ANTS 2 -m  PR[${R16_IMAGE},${R64_IMAGE},1,4] -r Gauss[3,0] -t SyN[0.5] -i 50x50x30 -o ${OUTPUT_PREFIX}.nii.gz --go-faster true )
  add_test(ANTS_CC_2_WARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2 ${R64_IMAGE} ${WARP_IMAGE} ${WARP}  -R ${R16_IMAGE} )
  add_test(ANTS_CC_2_JPG  ${TEST_BINARY_DIR}/ConvertToJpg ${WARP_IMAGE} ANTSCC2.jpg)
  add_test(ANTS_CC_2_WARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 11.7083 0.05)
  add_test(ANTS_CC_2_WARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.62 0.05)
  add_test(ANTS_CC_2_WARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.000461792 0.05)
  add_test(ANTS_CC_2_INVERSEWARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2 ${R16_IMAGE} ${INVERSEWARP_IMAGE} ${INVERSEWARP}   -R ${R16_IMAGE}  )
  add_test(ANTS_CC_2_INVERSEWARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 11.912 0.05)
  add_test(ANTS_CC_2_INVERSEWARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.62 0.05)
  add_test(ANTS_CC_2_INVERSEWARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.000442839 0.05)
  #add_test(ANTS_CC_2_INVERSEWARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -5 0.05)
  add_test(ANTS_CC_3   ${TEST_BINARY_DIR}/ANTS 2 -m  CC[${R16_IMAGE},${R64_IMAGE},1,4] -r Gauss[3,0] -t SyN[0.5] -i 50x50x30 -o ${OUTPUT_PREFIX}.nii.gz  --go-faster true )
  add_test(ANTS_CC_3_WARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2 ${R64_IMAGE} ${WARP_IMAGE} ${WARP}  -R ${R16_IMAGE} )
  add_test(ANTS_CC_3_JPG  ${TEST_BINARY_DIR}/ConvertToJpg ${WARP_IMAGE} ANTSCC2.jpg)
  add_test(ANTS_CC_3_WARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 11.7083 0.05)
  add_test(ANTS_CC_3_WARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.62 0.05)
  add_test(ANTS_CC_3_WARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.000461792 0.05)
  add_test(ANTS_CC_3_INVERSEWARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2 ${R16_IMAGE} ${INVERSEWARP_IMAGE} ${INVERSEWARP}   -R ${R16_IMAGE} )
  add_test(ANTS_CC_3_INVERSEWARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 11.912 0.05)
  add_test(ANTS_CC_3_INVERSEWARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.62 0.05)
  add_test(ANTS_CC_3_INVERSEWARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.000442839 0.05)
  #add_test(ANTS_CC_3_INVERSEWARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -5 0.05)
  add_test(ANTS_MSQ    ${TEST_BINARY_DIR}/ANTS 2 -m MSQ[${R16_IMAGE},${R64_IMAGE},1,0] -r Gauss[3,0] -t SyN[0.25] -i 50x50x30 -o ${OUTPUT_PREFIX}.nii.gz)
  add_test(ANTS_MSQ_WARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2 ${R64_IMAGE} ${WARP_IMAGE} ${WARP}  -R ${R16_IMAGE} )
  add_test(ANTS_MSQ_JPG  ${TEST_BINARY_DIR}/ConvertToJpg ${WARP_IMAGE} ANTSMSQ.jpg)
  add_test(ANTS_MSQ_WARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 11.7416 0.05)
  add_test(ANTS_MSQ_WARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.6 0.05)
  add_test(ANTS_MSQ_WARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -1.2 0.05)
  add_test(ANTS_MSQ_INVERSEWARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2 ${R16_IMAGE} ${INVERSEWARP_IMAGE} ${INVERSEWARP}   -R ${R16_IMAGE} )
  add_test(ANTS_MSQ_INVERSEWARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 11.7845 0.05)
  add_test(ANTS_MSQ_INVERSEWARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.6 0.05)
  add_test(ANTS_MSQ_INVERSEWARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -1.2 0.05)
  add_test(ANTS_MI_1   ${TEST_BINARY_DIR}/ANTS 2 -m  MI[${R16_IMAGE},${R64_IMAGE},1,32] -r Gauss[3,0] -t SyN[0.25] -i 50x50x30 -o ${OUTPUT_PREFIX}.nii.gz)
  add_test(ANTS_MI_1_WARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2 ${R64_IMAGE} ${WARP_IMAGE} ${WARP}  -R ${R16_IMAGE} )
  add_test(ANTS_MI_1_JPG  ${TEST_BINARY_DIR}/ConvertToJpg ${WARP_IMAGE} ANTSMI1.jpg)
  add_test(ANTS_MI_1_WARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 12.4 0.05)
  add_test(ANTS_MI_1_WARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.45 0.05)
  add_test(ANTS_MI_1_WARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.000370686 0.05)
  add_test(ANTS_MI_1_INVERSEWARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2 ${R16_IMAGE} ${INVERSEWARP_IMAGE} ${INVERSEWARP}  -R ${R16_IMAGE} )
  add_test(ANTS_MI_1_INVERSEWARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 12.6 0.05)
  add_test(ANTS_MI_1_INVERSEWARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.45 0.05)
  add_test(ANTS_MI_1_INVERSEWARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.000366879 0.05)
  add_test(ANTS_MI_2   ${TEST_BINARY_DIR}/ANTS 2 -m  MI[${R16_IMAGE},${R64_IMAGE},1,24] -r Gauss[3,0] -t SyN[0.25] -i 50x50x20 -o ${OUTPUT_PREFIX}.nii.gz)
  add_test(ANTS_MI_2_WARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2 ${R64_IMAGE} ${WARP_IMAGE} ${WARP}  -R ${R16_IMAGE} )
  add_test(ANTS_MI_2_JPG  ${TEST_BINARY_DIR}/ConvertToJpg ${WARP_IMAGE} ANTSMI2.jpg)
  add_test(ANTS_MI_2_WARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 12.4 0.05)
  add_test(ANTS_MI_2_WARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.45 0.05)
  add_test(ANTS_MI_2_WARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.000370686 0.05)
  add_test(ANTS_MI_2_INVERSEWARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2 ${R16_IMAGE} ${INVERSEWARP_IMAGE} ${INVERSEWARP}  -R ${R16_IMAGE}  )
  add_test(ANTS_MI_2_INVERSEWARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 12.6 0.05)
  add_test(ANTS_MI_2_INVERSEWARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.45 0.05)
  add_test(ANTS_MI_2_INVERSEWARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.000366879 0.05)
  ###
  #  ANTS transform testing
  ###
  add_test(ANTS_ELASTIC ${TEST_BINARY_DIR}/ANTS 2 -m PR[${R16_IMAGE},${R64_IMAGE},1,2] -t Elast[1]       -i 50x50x50 -r Gauss[0,1] -o ${OUTPUT_PREFIX}.nii.gz)
  add_test(ANTS_ELASTIC_WARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2 ${R64_IMAGE} ${WARP_IMAGE} ${WARP}   -R ${R16_IMAGE}  )
  add_test(ANTS_ELASTIC_JPG  ${TEST_BINARY_DIR}/ConvertToJpg ${WARP_IMAGE} ANTSELASTIC.jpg)
  add_test(ANTS_ELASTIC_WARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 12.5 0.05)
  add_test(ANTS_ELASTIC_WARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.4 0.05)
  add_test(ANTS_ELASTIC_WARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.000459869 0.05)
  add_test(ANTS_GSYN    ${TEST_BINARY_DIR}/ANTS 2 -m CC[${R16_IMAGE},${R64_IMAGE},1,3] -t SyN[0.25]       -i 50x50x50 -r Gauss[3,0.] -o ${OUTPUT_PREFIX}.nii.gz)
  add_test(ANTS_GSYN_WARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2 ${R64_IMAGE} ${WARP_IMAGE} ${WARP}  -R ${R16_IMAGE} )
  add_test(ANTS_GSYN_JPG  ${TEST_BINARY_DIR}/ConvertToJpg ${WARP_IMAGE} ANTSGSYN.jpg)
  add_test(ANTS_GSYN_WARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 11.7734 0.05)
  add_test(ANTS_GSYN_WARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.6 0.05)
  add_test(ANTS_GSYN_WARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.000478672 0.05)
  add_test(ANTS_GSYN_INVERSEWARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2 ${R16_IMAGE} ${INVERSEWARP_IMAGE} ${INVERSEWARP}  -R ${R16_IMAGE}  )
  add_test(ANTS_GSYN_JPGINV  ${TEST_BINARY_DIR}/ConvertToJpg ${INVERSEWARP_IMAGE} ANTSGSYNINV.jpg)
  add_test(ANTS_GSYN_INVERSEWARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 12.0541 0.05)
  add_test(ANTS_GSYN_INVERSEWARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.6 0.05)
  add_test(ANTS_GSYN_INVERSEWARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.000475175 0.05)
  add_test(ANTS_EXP     ${TEST_BINARY_DIR}/ANTS 2 -m PR[${R16_IMAGE},${R64_IMAGE},1,4] -t Exp[0.5,2,0.5]       -i 50x50x50 -r Gauss[0.5,0.25] -o ${OUTPUT_PREFIX}.nii.gz)
  add_test(ANTS_EXP_WARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2 ${R64_IMAGE} ${WARP_IMAGE} ${WARP}  -R ${R16_IMAGE} )
  add_test(ANTS_EXP_JPG  ${TEST_BINARY_DIR}/ConvertToJpg ${WARP_IMAGE} ANTSEXP.jpg)
  add_test(ANTS_EXP_WARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 12 0.05)
  add_test(ANTS_EXP_WARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.5 0.05)
  add_test(ANTS_EXP_WARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.000423693 0.05)
  add_test(ANTS_EXP_INVERSEWARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2 ${R16_IMAGE} ${INVERSEWARP_IMAGE} ${INVERSEWARP}   -R ${R16_IMAGE} )
  add_test(ANTS_EXP_INVERSEWARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 12.349 0.05)
  add_test(ANTS_EXP_INVERSEWARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.5 0.05)
  add_test(ANTS_EXP_INVERSEWARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.000339606 0.05)
  #add_test(ANTS_GSYN    ${TEST_BINARY_DIR}/ANTS 2 -m PR[${R16_IMAGE},${R64_IMAGE},1,2] -t SyN[0.75]            -i 50x50x50 -r Gauss[3,0.0,32] -o ${OUTPUT_PREFIX}.nii.gz)
  add_test(ANTS_SYN     ${TEST_BINARY_DIR}/ANTS 2 -m PR[${R16_IMAGE},${R64_IMAGE},1,2] -t SyN[0.5,2,0.05] -i 50x50x50 -r Gauss[3,0.0,32] -o ${OUTPUT_PREFIX}.nii.gz)
  add_test(ANTS_SYN_WARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2 ${R64_IMAGE} ${WARP_IMAGE} ${WARP}  -R ${R16_IMAGE}  )
  add_test(ANTS_SYN_JPG  ${TEST_BINARY_DIR}/ConvertToJpg ${WARP_IMAGE} ANTSSYN.jpg)
  add_test(ANTS_SYN_WARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 12.0239 0.05)
  add_test(ANTS_SYN_WARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.6 0.05)
  add_test(ANTS_SYN_WARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R16_IMAGE} ${WARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.000461922 0.05)
  add_test(ANTS_SYN_INVERSEWARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2 ${R16_IMAGE} ${INVERSEWARP_IMAGE} ${INVERSEWARP}  -R ${R16_IMAGE}  )
  add_test(ANTS_SYN_JPGINV  ${TEST_BINARY_DIR}/ConvertToJpg ${INVERSEWARP_IMAGE} ANTSSYNINV.jpg)
  add_test(ANTS_SYN_INVERSEWARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 12.5104 0.05)
  add_test(ANTS_SYN_INVERSEWARP_METRIC_1 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.6 0.05)
  add_test(ANTS_SYN_INVERSEWARP_METRIC_2 ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${R64_IMAGE} ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.000444279 0.05)
  ###
  # PSE sub-tests:  Check to see if .txt files and .vtk files also run correctly
  ###
  set(ANGEL_IMAGE_TXT ${DATA_DIR}/Smile.txt)
  set(DEVIL_IMAGE_TXT ${DATA_DIR}/Frown.txt)
  set(ANGEL_IMAGE_VTK ${DATA_DIR}/Smile.vtk)
  set(DEVIL_IMAGE_VTK ${DATA_DIR}/Frown.vtk)
  add_test(ANTS_PSE_MSQ_TXT ${TEST_BINARY_DIR}/ANTS 2 -i 191x170x155x40x30  -r Gauss[3,0] -t SyN[0.2]
    -m MSQ[${DEVIL_IMAGE},${ANGEL_IMAGE},1,0]
    -m PSE[${DEVIL_IMAGE},${ANGEL_IMAGE},${DEVIL_IMAGE_TXT},${ANGEL_IMAGE_TXT},1,0.1,11,1,1]
    --continue-affine 0 --number-of-affine-iterations 0
    -o ${OUTPUT_PREFIX}.nii.gz   --use-all-metrics-for-convergence 1 )
  add_test(ANTS_PSE_MSQ_VTK ${TEST_BINARY_DIR}/ANTS 2 -i 91x70x55x40x30  -r Gauss[3,0.,32] -t SyN[0.25]
    -m MSQ[${DEVIL_IMAGE},${ANGEL_IMAGE},1,0]
    -m PSE[${DEVIL_IMAGE},${ANGEL_IMAGE},${DEVIL_IMAGE_VTK},${ANGEL_IMAGE_VTK},1,0.33,11,1,25]
    --continue-affine 0 --number-of-affine-iterations 0
    -o ${OUTPUT_PREFIX}.nii.gz)
endif(NOT OLD_BASELINE_TESTS)
###
#  ANTS labeled data testing
###
option(RUN_LONG_TESTS "Run the time consuming tests." OFF )
if(RUN_LONG_TESTS)

add_test(ANTS_PSE_MSQ_IMG ${TEST_BINARY_DIR}/ANTS 2 -i 191x170x90x90x10  -r Gauss[6,0.25] -t SyN[1,2,0.1]
  -m PSE[${DEVIL_IMAGE},${ANGEL_IMAGE},${DEVIL_IMAGE},${ANGEL_IMAGE},0.25,0.1,100,0,10]
  -m MSQ[${DEVIL_IMAGE},${ANGEL_IMAGE},1,0.1]
  --continue-affine 0 --number-of-affine-iterations 0 --geodesic 2
  -o ${OUTPUT_PREFIX}.nii.gz)

add_test(ANTS_PSE_MSQ_IMG_WARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2
  ${ANGEL_IMAGE} ${WARP_IMAGE} ${WARP} -R  ${DEVIL_IMAGE} )

add_test(ANTS_PSE_JPG  ConvertToJpg ${WARP_IMAGE} ANTSPSE.jpg)

add_test(ANTS_PSE_MSQ_IMG_WARP_METRIC_0
  ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${DEVIL_IMAGE} ${WARP_IMAGE}
  ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 0.0116083 0.05)

add_test(ANTS_PSE_MSQ_IMG_WARP_METRIC_1
  ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${DEVIL_IMAGE} ${WARP_IMAGE}
  ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.884041 0.05)

add_test(ANTS_PSE_MSQ_IMG_WARP_METRIC_2
  ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${DEVIL_IMAGE} ${WARP_IMAGE}
  ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz -0.000710551 0.05)

add_test(ANTS_PSE_MSQ_IMG_INVERSEWARP
  ${TEST_BINARY_DIR}/WarpImageMultiTransform 2 ${DEVIL_IMAGE}
  ${INVERSEWARP_IMAGE} ${INVERSEWARP} -R  ${ANGEL_IMAGE}  )

add_test(ANTS_PSE_JPGINV  ${TEST_BINARY_DIR}/ConvertToJpg ${INVERSEWARP_IMAGE}
  ANTSPSEINV.jpg)

add_test(ANTS_PSE_MSQ_IMG_INVERSEWARP_METRIC_0
  ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 0 ${ANGEL_IMAGE}
  ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
  0.0109054 0.05)

add_test(ANTS_PSE_MSQ_IMG_INVERSEWARP_METRIC_1
  ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 1 ${ANGEL_IMAGE}
  ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
  -0.901632 0.05)

add_test(ANTS_PSE_MSQ_IMG_INVERSEWARP_METRIC_2
  ${TEST_BINARY_DIR}/MeasureImageSimilarity 2 2 ${ANGEL_IMAGE}
  ${INVERSEWARP_IMAGE} ${OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz
  -0.000704717 0.05)
###
#  ANTS images with non-trival rotation header test
###
#set(ROT_REF_IMAGE ${DATA_DIR}/ref2.nii.gz)
#set(ROT_MOV_IMAGE ${DATA_DIR}/mov2.nii.gz)
set(ROT_REF_IMAGE ${DATA_DIR}/r16roth.nii.gz)
set(ROT_MOV_IMAGE ${DATA_DIR}/r64roth.nii.gz)
set(ROT_OUTPUT_PREFIX ${CMAKE_BINARY_DIR}/RotTEST)
set(ROT_WARP ${ROT_OUTPUT_PREFIX}Warp.nii.gz ${ROT_OUTPUT_PREFIX}Affine.txt )
set(ROT_INVERSEWARP -i ${ROT_OUTPUT_PREFIX}Affine.txt ${ROT_OUTPUT_PREFIX}InverseWarp.nii.gz )
set(ROT_WARP_IMAGE ${CMAKE_BINARY_DIR}/rotwarped.nii.gz)
set(ROT_INVERSEWARP_IMAGE ${CMAKE_BINARY_DIR}/rotinversewarped.nii.gz)
set(ROT_WARP_FILES ${ROT_OUTPUT_PREFIX}Warp*vec.nii.gz ${ROT_OUTPUT_PREFIX}InverseWarp*vec.nii.gz ${ROT_OUTPUT_PREFIX}Affine.txt )

#add_test(ANTS_ROT_SYN ${TEST_BINARY_DIR}/ANTS 3 -m MSQ[${ROT_REF_IMAGE},${ROT_MOV_IMAGE},1,0] -t SyN[0.5,2,0.05] -i 50x5x0 -r Gauss[3,0.] -o ${ROT_OUTPUT_PREFIX}.nii.gz)
#add_test(ANTS_ROT_SYN_WARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 3 ${ROT_MOV_IMAGE} ${ROT_WARP_IMAGE} -R ${ROT_REF_IMAGE} ${ROT_WARP} )
#add_test(ANTS_ROT_SYN_WARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 3 0 ${ROT_REF_IMAGE} ${ROT_WARP_IMAGE} ${ROT_OUTPUT_PREFIX}log.txt ${ROT_OUTPUT_PREFIX}metric.nii.gz 35.4473 0.5)
#add_test(ANTS_ROT_SYN_INVERSEWARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 3 ${ROT_REF_IMAGE} ${ROT_INVERSEWARP_IMAGE} -R ${ROT_MOV_IMAGE} ${ROT_INVERSEWARP})
#add_test(ANTS_ROT_SYN_INVERSEWARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 3 0 ${ROT_MOV_IMAGE} ${ROT_INVERSEWARP_IMAGE} ${ROT_OUTPUT_PREFIX}log.txt ${OUTPUT_PREFIX}metric.nii.gz 34.8184 0.5)
#add_test(ANTS_ROT_SYN_CLEAN rm ${ROT_WARP_FILES})

add_test(ANTS_ROT_GSYN ${TEST_BINARY_DIR}/ANTS 3 -m
  MSQ[${ROT_REF_IMAGE},${ROT_MOV_IMAGE},1,0] -t SyN[0.25] -i 50x5x0 -r
  Gauss[3,0.0,32] -o ${ROT_OUTPUT_PREFIX}.nii.gz)

add_test(ANTS_ROT_GSYN_WARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 3
  ${ROT_MOV_IMAGE} ${ROT_WARP_IMAGE} -R ${ROT_REF_IMAGE} ${ROT_WARP} )

add_test(ANTS_ROT_GSYN_WARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity
  3 0 ${ROT_REF_IMAGE} ${ROT_WARP_IMAGE} ${ROT_OUTPUT_PREFIX}log.txt
  ${ROT_OUTPUT_PREFIX}metric.nii.gz 35.4473 0.5)

add_test(ANTS_ROT_GSYN_INVERSEWARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 3
  ${ROT_REF_IMAGE} ${ROT_INVERSEWARP_IMAGE} -R ${ROT_MOV_IMAGE}
  ${ROT_INVERSEWARP})

add_test(ANTS_ROT_GSYN_INVERSEWARP_METRIC_0
  ${TEST_BINARY_DIR}/MeasureImageSimilarity 3 0 ${ROT_MOV_IMAGE}
  ${ROT_INVERSEWARP_IMAGE} ${ROT_OUTPUT_PREFIX}log.txt
  ${OUTPUT_PREFIX}metric.nii.gz  34.8184 0.5)

#add_test(ANTS_ROT_GSYN_CLEAN rm ${ROT_WARP_FILES})

add_test(ANTS_ROT_EXP ${TEST_BINARY_DIR}/ANTS 3 -m
  MSQ[${ROT_REF_IMAGE},${ROT_MOV_IMAGE},1,0] -t Exp[3,2] -i 50x5x1 -r
  Gauss[3,0.0,32] -o ${ROT_OUTPUT_PREFIX}.nii.gz)

add_test(ANTS_ROT_EXP_WARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 3
  ${ROT_MOV_IMAGE} ${ROT_WARP_IMAGE} -R ${ROT_REF_IMAGE} ${ROT_WARP} )

add_test(ANTS_ROT_EXP_WARP_METRIC_0 ${TEST_BINARY_DIR}/MeasureImageSimilarity 3
  0 ${ROT_REF_IMAGE} ${ROT_WARP_IMAGE} ${ROT_OUTPUT_PREFIX}log.txt
  ${ROT_OUTPUT_PREFIX}metric.nii.gz 35.4473 0.5)

add_test(ANTS_ROT_EXP_WARP2 ${TEST_BINARY_DIR}/WarpImageMultiTransform 3
  ${ROT_MOV_IMAGE} ${ROT_WARP_IMAGE} -R ${ROT_REF_IMAGE} --ANTS-prefix
  ${ROT_OUTPUT_PREFIX} )

add_test(ANTS_ROT_EXP_WARP2_METRIC_0_2
  ${TEST_BINARY_DIR}/MeasureImageSimilarity 3 0 ${ROT_REF_IMAGE}
  ${ROT_WARP_IMAGE} ${ROT_OUTPUT_PREFIX}log.txt
  ${ROT_OUTPUT_PREFIX}metric.nii.gz 35.4473 0.5)

add_test(ANTS_ROT_EXP_INVERSEWARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 3
  ${ROT_REF_IMAGE} ${ROT_INVERSEWARP_IMAGE} -R ${ROT_MOV_IMAGE} ${ROT_INVERSEWARP})

add_test(ANTS_ROT_EXP_INVERSEWARP_METRIC_0
  ${TEST_BINARY_DIR}/MeasureImageSimilarity 3 0 ${ROT_MOV_IMAGE}
  ${ROT_INVERSEWARP_IMAGE} ${ROT_OUTPUT_PREFIX}log.txt
  ${OUTPUT_PREFIX}metric.nii.gz  34.8184 0.5)

add_test(ANTS_ROT_EXP_INVERSEWARP2 ${TEST_BINARY_DIR}/WarpImageMultiTransform 3
  ${ROT_REF_IMAGE} ${ROT_INVERSEWARP_IMAGE} -R ${ROT_MOV_IMAGE}
  --ANTS-prefix-invert ${ROT_OUTPUT_PREFIX})

add_test(ANTS_ROT_EXP_INVERSEWARP2_METRIC_0_2
  ${TEST_BINARY_DIR}/MeasureImageSimilarity 3 0 ${ROT_MOV_IMAGE}
  ${ROT_INVERSEWARP_IMAGE} ${ROT_OUTPUT_PREFIX}log.txt
  ${OUTPUT_PREFIX}metric.nii.gz  34.8184 0.5)

#add_test(ANTS_ROT_EXP_CLEAN rm ${ROT_WARP_FILES})
###
#  Test SyN with time
###
set(CHALF_IMAGE ${DATA_DIR}/chalf.nii.gz)
set(C_IMAGE ${DATA_DIR}/c.nii.gz)

add_test(ANTS_SYN_WITH_TIME ${TEST_BINARY_DIR}/ANTS 2 -m
  MSQ[${CHALF_IMAGE},${C_IMAGE},1,0.] -t SyN[1,10,0.05] -i 150x100x2x2 -r
  Gauss[0.5,0.1] -o ${OUTPUT_PREFIX} --geodesic 1 --number-of-affine-iterations 0)

add_test(ANTS_SYN_WITH_TIME_WARP ${TEST_BINARY_DIR}/WarpImageMultiTransform 2
  ${C_IMAGE} ${OUTPUT_PREFIX}.nii.gz ${OUTPUT_PREFIX}Warp.nii.gz  -R
  ${CHALF_IMAGE} )

add_test(ANTS_SYN_WITH_TIME_METRIC ${TEST_BINARY_DIR}/MeasureImageSimilarity 2
  0 ${CHALF_IMAGE} ${OUTPUT_PREFIX}.nii.gz ${OUTPUT_PREFIX}log.txt
  ${OUTPUT_PREFIX}metric.nii.gz  0.0943736 0.1)
##
# Apocrita tests
##
set(R16_MASK ${DATA_DIR}/r16mask.nii.gz)
set(R16_PRIORS ${DATA_DIR}/r16priors.nii.gz)

#add_test(APOC_OTSU_INIT ${TEST_BINARY_DIR}/Apocrita 2 -i otsu[${R16_IMAGE},3] -x ${R16_MASK} -n 10 -m [0.3,1x1,0.2,0.1] -o ${OUTPUT_PREFIX}.nii.gz )
#add_test(APOC_OTSU_INIT_RADIUS_2x2 ${TEST_BINARY_DIR}/Apocrita 2 -i otsu[${R16_IMAGE},3] -x ${R16_MASK} -n 10 -m [0.3,2,0.2,0.1] -o ${OUTPUT_PREFIX}.nii.gz )
#add_test(APOC_KMEANS_INIT ${TEST_BINARY_DIR}/Apocrita 2 -i kmeans[${R16_IMAGE},3] -x ${R16_MASK} -n 10 -m [0.3,1x1,0.2,0.1] -o [${OUTPUT_PREFIX}.nii.gz,${OUTPUT_PREFIX}_posteriors%d.nii.gz])
#add_test(APOC_PRIORLABELIMAGE_INIT ${TEST_BINARY_DIR}/Apocrita 2 -i priorlabelimage[${R16_IMAGE},5,${R16_PRIORS},0.5] -x ${R16_MASK} -n 10 -m [0.3,1x1,0.2,0.1] -o [${OUTPUT_PREFIX}.nii.gz,${OUTPUT_PREFIX}_posteriors%d.nii.gz] -l 1[1,0.75] -l 2[1,1.0] -l 3[0.5,0.5] -l 4[1,1])
endif(RUN_LONG_TESTS)
