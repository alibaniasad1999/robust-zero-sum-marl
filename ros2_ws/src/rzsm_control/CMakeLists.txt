cmake_minimum_required(VERSION 3.18)
project(rzsm_control)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── ROS 2 dependencies ───────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)

# ── LibTorch ─────────────────────────────────────────────────────
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ── Existing rzsm_lib from cpp/ build ────────────────────────────
# Set RZSM_LIB_DIR to the cpp/ build directory containing librzsm_lib.a
# Default: assumes standard project layout (ros2_ws/src/rzsm_control -> ../../../cpp/build)
set(RZSM_LIB_DIR "" CACHE PATH "Path to cpp/ build directory containing librzsm_lib.a")
if(NOT RZSM_LIB_DIR)
    set(RZSM_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../cpp/build")
endif()

set(RZSM_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../cpp/include")

find_library(RZSM_LIB rzsm_lib HINTS "${RZSM_LIB_DIR}" REQUIRED)
message(STATUS "Found rzsm_lib: ${RZSM_LIB}")

# ── MuJoCo (transitive dependency of rzsm_lib) ───────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cpp/cmake")
find_package(MuJoCo REQUIRED)

# ── cnpy (transitive dependency of rzsm_lib) ─────────────────────
# cnpy is built inside cpp/build via FetchContent; link its library
find_library(CNPY_LIB cnpy HINTS "${RZSM_LIB_DIR}/_deps/cnpy-build" REQUIRED)
find_path(CNPY_INCLUDE cnpy.h HINTS "${RZSM_LIB_DIR}/_deps/cnpy-src" REQUIRED)

# ── Generate custom messages ──────────────────────────────────────
rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/DetectorStatus.msg"
    DEPENDENCIES builtin_interfaces
)

# ── Controller node executable ────────────────────────────────────
add_executable(robust_controller_node
    src/robust_controller_node.cpp
    src/main.cpp
)

target_include_directories(robust_controller_node PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${RZSM_INCLUDE_DIR}
    ${CNPY_INCLUDE}
)

target_link_libraries(robust_controller_node
    ${RZSM_LIB}
    ${TORCH_LIBRARIES}
    ${CNPY_LIB}
    MuJoCo::MuJoCo
)

ament_target_dependencies(robust_controller_node
    rclcpp
    sensor_msgs
    geometry_msgs
    diagnostic_msgs
    std_msgs
)

# Link generated message typesupport to the node
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(robust_controller_node "${cpp_typesupport_target}")

# ── Install ───────────────────────────────────────────────────────
install(TARGETS robust_controller_node
    DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
    DESTINATION include)

install(DIRECTORY launch config
    DESTINATION share/${PROJECT_NAME})

install(PROGRAMS scripts/export_onnx.py
    DESTINATION lib/${PROJECT_NAME})

ament_package()
