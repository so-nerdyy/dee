cmake_minimum_required(VERSION 3.18)

project(dee.cpp VERSION 0.2.0 DESCRIPTION "Dynamic Expert Eviction inference engine" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(DEE_CUDA "Enable the CUDA runtime and kernels" OFF)
option(DEE_BUILD_TESTS "Build unit tests" ON)
option(DEE_BUILD_GGML "Build the vendored ggml reference backend" OFF)
option(DEE_CUDA_VALIDATE "Synchronize CUDA launches close to their call site" OFF)

if(DEE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Do not overwrite a caller's architecture selection.  T4 users should
    # configure -DCMAKE_CUDA_ARCHITECTURES=75; 75 is also our portable default.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING "CUDA architectures to build for")
    endif()
endif()

set(DEE_CORE_SOURCES
    src/dee.cpp
    src/json_min.cpp
    src/profiling.cpp
    src/weight_mmap.cpp
    src/vram_cache.cpp
    src/async_prefetcher.cpp
    src/pt_loader.cpp
    src/oracle.cpp
    src/engine.cpp)

add_library(dee_core STATIC ${DEE_CORE_SOURCES})
set_target_properties(dee_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(dee_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(dee_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>)

find_package(ZLIB REQUIRED)
target_link_libraries(dee_core PUBLIC ZLIB::ZLIB)

if(DEE_CUDA)
    target_sources(dee_core PRIVATE src/cuda_convert.cu src/swiglu_cuda.cu)
    target_compile_definitions(dee_core PUBLIC DEE_CUDA=1)
    if(DEE_CUDA_VALIDATE)
        target_compile_definitions(dee_core PUBLIC DEE_CUDA_VALIDATE=1)
    endif()
    target_link_libraries(dee_core PUBLIC CUDA::cudart CUDA::cublas)
    set_target_properties(dee_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

if(DEE_BUILD_GGML)
    set(GGML_CUDA ${DEE_CUDA} CACHE BOOL "ggml CUDA backend" FORCE)
    set(GGML_METAL OFF CACHE BOOL "" FORCE)
    set(GGML_BLAS OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)
    target_include_directories(dee_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/include)
    target_link_libraries(dee_core PUBLIC ggml)
    if(DEE_CUDA)
        target_link_libraries(dee_core PUBLIC ggml-cuda)
    endif()
endif()

add_executable(dee_cli main.cpp)
target_link_libraries(dee_cli PRIVATE dee_core)

if(DEE_BUILD_TESTS)
    enable_testing()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_custom_target(dee_test_assets
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/gen_synthetic_shard.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_shard.safetensors
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/gen_moe_shard.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/ornith_moe256.safetensors
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/gen_split_router_shards.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/split-router
        BYPRODUCTS
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_shard.safetensors
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/ornith_moe256.safetensors
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/split-router/split-a.safetensors
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/split-router/split-b.safetensors
        COMMENT "Generating deterministic safetensors test shards")
    set(DEE_TEST_SOURCES
        tests/test_smoke.cpp
        tests/test_weight_mmap.cpp
        tests/test_vram_cache.cpp
        tests/test_async_prefetcher.cpp
        tests/test_oracle.cpp
        tests/test_engine.cpp)
    list(APPEND DEE_TEST_SOURCES tests/test_real_router.cpp)
    if(DEE_CUDA)
        list(APPEND DEE_TEST_SOURCES tests/test_swiglu_cuda.cpp)
    endif()
    foreach(test_source IN LISTS DEE_TEST_SOURCES)
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source})
        target_link_libraries(${test_name} PRIVATE dee_core)
        add_dependencies(${test_name} dee_test_assets)
        add_test(NAME ${test_name} COMMAND ${test_name})
        set_tests_properties(${test_name} PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endforeach()
    add_test(
        NAME test_ornith_support
        COMMAND ${Python3_EXECUTABLE} -m unittest tests/test_ornith_support.py)
    set_tests_properties(test_ornith_support PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

add_custom_target(lightning_t4_benchmark
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/scripts/setup_lightning_t4.sh --benchmark
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    USES_TERMINAL
    COMMENT "Prepare assets, build sm_75 CUDA binary, and run the DEE benchmark")
