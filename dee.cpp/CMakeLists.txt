cmake_minimum_required(VERSION 3.18)

project(dee.cpp
    VERSION 0.1.0
    DESCRIPTION "Dynamic Expert Eviction inference engine"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(DEE_CUDA        "Enable CUDA GPU offload (RTX 3090 / sm_86)" OFF)
option(DEE_BUILD_TESTS "Build unit tests"                          ON)
option(DEE_BUILD_GGML  "Build + link the vendored ggml backend"   OFF)

# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------
# DEE_CUDA defaults OFF so the engine builds on CPU-only machines.
# On the RTX 3090 box, configure with -DDEE_CUDA=ON (requires CUDA Toolkit).
if(DEE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    # Broad consumer/datacenter compatibility: Pascal (sm_60) through Blackwell
    # (sm_90). The engine logic is hardware-agnostic; ggml-cuda ships native
    # kernels for all of these. (Production target: RTX 3090 = sm_86.)
    set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;89;90" CACHE STRING "CUDA architectures" FORCE)
    add_compile_definitions(DEE_CUDA=1)
endif()

# ---------------------------------------------------------------------------
# Vendored ggml (foundation / reference)
# ---------------------------------------------------------------------------
# ggml is vendored (see third_party/ggml + third_party/README.md) and is the
# tensor/compute backend for the FORWARD PASS (later steps). Step 4 is the
# data layer only (mmap + resolver) and does NOT link ggml, so the project
# builds without it. Flip DEE_BUILD_GGML=ON when the forward pass is added.
option(DEE_BUILD_GGML "Build + link the vendored ggml backend" OFF)
if(DEE_BUILD_GGML)
    # Drive ggml's own backend options from ours. ggml-cuda only builds when
    # DEE_CUDA is ON AND the CUDA Toolkit is present.
    set(GGML_CUDA ${DEE_CUDA} CACHE BOOL "ggml: CUDA backend" FORCE)
    set(GGML_METAL OFF CACHE BOOL "" FORCE)
    set(GGML_BLAS   OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)
    set(GGML_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/include)
    set(GGML_LIB ggml)
    if(DEE_CUDA)
        set(GGML_CUDA_LIB ggml-cuda)
    endif()
endif()

# ---------------------------------------------------------------------------
# dee.cpp core library
# ---------------------------------------------------------------------------
file(GLOB DEE_SRC CONFIGURE_DEPENDS src/*.cpp)
add_library(dee_core STATIC ${DEE_SRC})
target_include_directories(dee_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
find_package(ZLIB REQUIRED)
target_link_libraries(dee_core PUBLIC ZLIB::ZLIB)
if(DEE_CUDA)
    # Compile the real CUDA SwiGLU kernels with nvcc and link the CUDA runtime.
    file(GLOB DEE_CUDA_SRC CONFIGURE_DEPENDS src/*.cu)
    target_sources(dee_core PRIVATE ${DEE_CUDA_SRC})
    target_link_libraries(dee_core PUBLIC CUDA::cudart)
endif()
if(DEE_BUILD_GGML)
    target_include_directories(dee_core PUBLIC ${GGML_INCLUDE_DIR})
    target_link_libraries(dee_core PUBLIC ${GGML_LIB})
    if(DEE_CUDA)
        target_link_libraries(dee_core PUBLIC ${GGML_CUDA_LIB} CUDA::cudart)
    endif()
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(DEE_BUILD_TESTS)
    enable_testing()
    file(GLOB DEE_TESTS CONFIGURE_DEPENDS tests/*.cpp)
    foreach(t ${DEE_TESTS})
        get_filename_component(tname ${t} NAME_WE)
        add_executable(${tname} ${t})
        target_link_libraries(${tname} PRIVATE dee_core)
        add_test(NAME ${tname} COMMAND ${tname})
        # tests reference repo-relative paths (tests/data/...); run from source root
        set_tests_properties(${tname} PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endforeach()
endif()

# ---------------------------------------------------------------------------
# CLI benchmark driver (Step 8)
# ---------------------------------------------------------------------------
add_executable(dee_cli main.cpp)
target_link_libraries(dee_cli PRIVATE dee_core)
