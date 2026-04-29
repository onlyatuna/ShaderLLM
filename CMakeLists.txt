cmake_minimum_required(VERSION 3.22)

project(CHIMERA_V LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(Vulkan REQUIRED)

# --- CHIMERA-V Shader 雙重編譯引擎 (Dual Pipeline) ---
set(SLANG_PATH "${CMAKE_CURRENT_SOURCE_DIR}/raw/slang-2026.7-windows-x86_64")
set(SLANGC_EXECUTABLE "${SLANG_PATH}/bin/slangc.exe")

# 尋找 Vulkan SDK 的 glslangValidator
if(DEFINED ENV{VULKAN_SDK})
    set(GLSLANG_EXECUTABLE "$ENV{VULKAN_SDK}/Bin/glslangValidator.exe")
else()
    set(GLSLANG_EXECUTABLE "glslangValidator") # 依賴系統 PATH
endif()

# --- 全局輸出配置 (Best Practice) ---
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(SHADER_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/src/shaders")
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

set(SHADER_SOURCES 
    src/shaders/nca_evolve.comp
    src/shaders/nca_uhd.comp
    src/shaders/nca_tanh.comp
    src/shaders/nca_l0.comp
    src/shaders/verify_flow.comp
    src/shaders/coopmat_test.comp
    src/shaders/nca_spatial_agg.slang
    src/shaders/nca_decode.slang
    src/shaders/nca_inject.slang
    src/shaders/nca_ple_gate.slang
)

set(SHADER_SPVS "")
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    get_filename_component(SHADER_EXT ${SHADER} EXT)
    # 修正：直接輸出到執行環境目錄
    set(SPV_OUTPUT "${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")

    if("${SHADER_NAME}" MATCHES "^(nca_evolve|coopmat_test)$")
        # 核心 Tensor Core 路徑 (保留 15 TFLOPS 絕對控制權，使用 glslangValidator)
        add_custom_command(
            OUTPUT ${SPV_OUTPUT}
            COMMAND ${GLSLANG_EXECUTABLE} -V --target-env vulkan1.3 ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER} -o ${SPV_OUTPUT}
            DEPENDS ${SHADER}
            COMMENT "CHIMERA-V: Compiling GLSL (Tensor Core) via glslangValidator -> ${SPV_OUTPUT}"
            VERBATIM
        )
    else()
        # 常規著色器路徑 (繼續享受 Slang 的暫存器分配與優化紅利)
        set(LANG_FLAG "")
        set(PROFILE_STR "spirv_1_5")
        if("${SHADER_EXT}" STREQUAL ".comp")
            set(LANG_FLAG "-allow-glsl")
            set(PROFILE_STR "glsl_460")
        endif()

        add_custom_command(
            OUTPUT ${SPV_OUTPUT}
            COMMAND ${SLANGC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER} 
                    ${LANG_FLAG}
                    -I ${SLANG_PATH}/bin
                    -I ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders
                    -target spirv -o ${SPV_OUTPUT}
                    -profile ${PROFILE_STR}
                    -O3
            DEPENDS ${SHADER}
            COMMENT "CHIMERA-V: Optimizing Shader (${SHADER_EXT}) via Slang -> ${SPV_OUTPUT}"
            VERBATIM
        )
    endif()
    list(APPEND SHADER_SPVS ${SPV_OUTPUT})
endforeach()

# --- 全局資源管理 ---
add_custom_target(chimera_shaders ALL DEPENDS ${SHADER_SPVS})
add_custom_target(chimera_assets ALL)
add_dependencies(chimera_assets chimera_shaders)

# 自動複製權重檔案
set(WEIGHTS_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/weights")
add_custom_command(
    TARGET chimera_assets PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${WEIGHTS_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory 
            "${CMAKE_CURRENT_SOURCE_DIR}/weights" 
            ${WEIGHTS_OUTPUT_DIR}
    COMMENT "CHIMERA-V: Syncing neural weights to execution environment..."
)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(chimera_v_opt src/main_opt.cpp src/engine/NcaKernel.cu)
add_dependencies(chimera_v_opt chimera_assets) 

add_executable(chimera_tma src/main_tma.cpp src/engine/NcaKernel.cu)
add_dependencies(chimera_tma chimera_assets)

add_executable(chimera_cuda src/main_cuda.cpp src/engine/NcaKernel.cu)
add_executable(chimera_uhd  src/main_uhd.cpp src/engine/NcaKernel.cu)
add_dependencies(chimera_uhd chimera_assets)

add_executable(uhd_inspector src/utils/check_uhd_specs.cpp)
add_executable(rtx_inspector src/utils/check_rtx_specs.cpp)
add_executable(vulkan_limits_check src/utils/check_vulkan_limits.cpp)
add_executable(l0_inspector src/utils/check_l0_specs.cpp)
add_executable(chimera_l0 src/main_l0.cpp)
add_dependencies(chimera_l0 chimera_assets)

add_executable(verify_data_flow src/utils/verify_data_flow.cpp)
add_dependencies(verify_data_flow chimera_assets)

add_executable(verify_ple_table src/utils/verify_ple_table.cpp)
add_dependencies(verify_ple_table chimera_assets)

# 連結庫
target_link_libraries(chimera_v_opt PRIVATE CUDA::cudart CUDA::cublas Vulkan::Vulkan)
target_link_libraries(chimera_tma PRIVATE CUDA::cudart CUDA::cublas Vulkan::Vulkan)
target_link_libraries(chimera_cuda PRIVATE CUDA::cudart CUDA::cublas)
target_link_libraries(chimera_uhd PRIVATE CUDA::cudart CUDA::cublas Vulkan::Vulkan)
target_link_libraries(uhd_inspector PRIVATE Vulkan::Vulkan)
target_link_libraries(vulkan_limits_check PRIVATE Vulkan::Vulkan)
target_link_libraries(rtx_inspector PRIVATE CUDA::cudart Vulkan::Vulkan)
target_include_directories(verify_data_flow PRIVATE src)
target_link_libraries(verify_data_flow PRIVATE Vulkan::Vulkan)

# Level Zero 目標配置
target_include_directories(l0_inspector PRIVATE ${LEVEL_ZERO_INCLUDE_DIR})
target_link_libraries(l0_inspector PRIVATE ${LEVEL_ZERO_LIBRARY})
target_include_directories(chimera_l0 PRIVATE ${LEVEL_ZERO_INCLUDE_DIR})
target_link_libraries(chimera_l0 PRIVATE ${LEVEL_ZERO_LIBRARY})

# SYCL 目標配置 (暫時停用)
# target_compile_options(chimera_sycl PRIVATE /fsycl /Qmkl)
# target_link_options(chimera_sycl PRIVATE /fsycl /Qmkl)
