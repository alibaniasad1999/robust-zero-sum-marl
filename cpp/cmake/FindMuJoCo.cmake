# FindMuJoCo.cmake
# Locates MuJoCo >= 3.0 installation.
# Set MUJOCO_DIR or environment variable to the install prefix.
#
# Provides: MuJoCo::MuJoCo imported target

find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h
    HINTS ${MUJOCO_DIR} ENV MUJOCO_DIR
    PATH_SUFFIXES include
)

find_library(MUJOCO_LIBRARY
    NAMES mujoco
    HINTS ${MUJOCO_DIR} ENV MUJOCO_DIR
    PATH_SUFFIXES lib lib64 bin
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MuJoCo DEFAULT_MSG
    MUJOCO_INCLUDE_DIR MUJOCO_LIBRARY)

if(MuJoCo_FOUND AND NOT TARGET MuJoCo::MuJoCo)
    add_library(MuJoCo::MuJoCo SHARED IMPORTED)
    set_target_properties(MuJoCo::MuJoCo PROPERTIES
        IMPORTED_LOCATION "${MUJOCO_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_INCLUDE_DIR}"
    )
endif()
