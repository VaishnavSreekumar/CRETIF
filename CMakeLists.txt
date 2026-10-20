cmake_minimum_required(VERSION 3.24)
project(CuInfer LANGUAGES CXX CUDA)

# ── C++ Standard ──────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── CUDA Standard ─────────────────────────────────────────────
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default to SM 8.6 (RTX 3060 Ampere) if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# ── CUDA Toolkit ──────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# Suppress warnings from nvcc-generated device-link stubs (prelinked_fatbinc).
# These are in auto-generated .reg.c files we don't control; CMAKE_CUDA_FLAGS
# applies to all nvcc invocations including device-link.
if(MSVC)
    string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=/wd4100 -Xcompiler=/wd4505")
endif()

# ── Compiler Warnings ─────────────────────────────────────────
# Use generator expressions so MSVC flags only apply to CXX (not passed raw to nvcc).
# CUDA warnings are handled separately on the cuinfer_lib target below.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
elseif(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/WX>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        $<$<COMPILE_LANGUAGE:CXX>:/external:anglebrackets>
        $<$<COMPILE_LANGUAGE:CXX>:/external:W0>
    )
endif()

# ── Dependencies ──────────────────────────────────────────────
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ── CUDA Error Checking Macro ─────────────────────────────────
# Defined in include/cuinfer/cuda_utils.h

# ── Library ───────────────────────────────────────────────────
file(GLOB_RECURSE CUINFER_SOURCES
    "src/*.cpp"
    "src/*.cu"
)
# Exclude main.cu from the library — it contains main() which would
# conflict with GTest's main() when linking tests.
list(FILTER CUINFER_SOURCES EXCLUDE REGEX "src/main\\.(cpp|cu)$")

add_library(cuinfer_lib ${CUINFER_SOURCES})
target_include_directories(cuinfer_lib PUBLIC include)
target_link_libraries(cuinfer_lib PUBLIC CUDA::cudart)

# ── CUDA-specific MSVC warning flags ─────────────────────────
# Applied to all our targets (not GoogleTest, which has no .cu files).
# Suppressions are for warnings triggered by CUDA runtime headers and
# nvcc-generated device-link code, not by our source code:
#   C4100 — unreferenced parameter (CUDA runtime / device-link stubs)
#   C4201 — nameless struct/union (CUB headers)
#   C4505 — unreferenced internal function (__cudaUnregisterBinaryUtil)
function(cuinfer_set_cuda_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
                -Xcompiler=/W4 -Xcompiler=/WX
                -Xcompiler=/external:anglebrackets -Xcompiler=/external:W0
                -Xcompiler=/wd4100
                -Xcompiler=/wd4201
                -Xcompiler=/wd4505
            >
        )
    endif()
endfunction()

cuinfer_set_cuda_warnings(cuinfer_lib)

# Enable separable compilation for device code linking.
# CUDA_RESOLVE_DEVICE_SYMBOLS ON on the library ensures that device-link
# symbols (__cudaRegisterLinkedBinary_*) are resolved inside the .lib itself,
# so downstream executables don't need their own device-link step.
set_target_properties(cuinfer_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# ── Main Executable ───────────────────────────────────────────
add_executable(cuinfer src/main.cu)
target_link_libraries(cuinfer PRIVATE cuinfer_lib)
set_target_properties(cuinfer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
cuinfer_set_cuda_warnings(cuinfer)

# ── Tests ─────────────────────────────────────────────────────
enable_testing()

file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp" "tests/*.cu")
if(TEST_SOURCES)
    add_executable(cuinfer_tests ${TEST_SOURCES})
    target_link_libraries(cuinfer_tests PRIVATE cuinfer_lib GTest::gtest_main)
    set_target_properties(cuinfer_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    cuinfer_set_cuda_warnings(cuinfer_tests)
    # Suppress nvcc #177-D (member declared but never referenced) triggered by
    # GoogleTest TEST() macros generating unused test_info_ members in .cu files.
    target_compile_options(cuinfer_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=177>)

    include(GoogleTest)
    gtest_discover_tests(cuinfer_tests)
endif()

# ── Benchmarks ────────────────────────────────────────────────
file(GLOB_RECURSE BENCH_SOURCES "benchmarks/*.cpp" "benchmarks/*.cu")
foreach(bench_src ${BENCH_SOURCES})
    get_filename_component(bench_name ${bench_src} NAME_WE)
    add_executable(${bench_name} ${bench_src})
    target_link_libraries(${bench_name} PRIVATE cuinfer_lib)
    set_target_properties(${bench_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
    cuinfer_set_cuda_warnings(${bench_name})
    # Link cuBLAS only for matmul benchmark (comparison against vendor library)
    if(bench_name STREQUAL "bench_matmul")
        target_link_libraries(${bench_name} PRIVATE CUDA::cublas)
    endif()
endforeach()
