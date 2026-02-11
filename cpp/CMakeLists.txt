cmake_minimum_required(VERSION 3.18)
project(robust_zero_sum_marl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── LibTorch ──────────────────────────────────────────────────────
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ── MuJoCo ────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(MuJoCo REQUIRED)

# ── cnpy (for .npz file writing) ─────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    cnpy
    GIT_REPOSITORY https://github.com/rogersce/cnpy.git
    GIT_TAG master
)
FetchContent_MakeAvailable(cnpy)

# ── Assets path define ───────────────────────────────────────────
set(RZSM_ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/assets")
add_compile_definitions(RZSM_ASSETS_DIR="${RZSM_ASSETS_DIR}")

# ── Main library ─────────────────────────────────────────────────
file(GLOB_RECURSE LIB_SOURCES
    "src/networks/*.cpp"
    "src/agents/*.cpp"
    "src/detector/*.cpp"
    "src/envs/*.cpp"
    "src/utils/*.cpp"
)

add_library(rzsm_lib STATIC ${LIB_SOURCES})
target_include_directories(rzsm_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(rzsm_lib PUBLIC
    ${TORCH_LIBRARIES}
    MuJoCo::MuJoCo
    cnpy
)

# ── Executable ───────────────────────────────────────────────────
add_executable(train src/main.cpp)
target_link_libraries(train PRIVATE rzsm_lib)

# ── Tests ────────────────────────────────────────────────────────
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
