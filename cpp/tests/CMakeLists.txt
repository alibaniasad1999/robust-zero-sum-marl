include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

set(TEST_SOURCES
    test_mlp.cpp
    test_replay_buffer.cpp
    test_detector.cpp
    test_logger.cpp
)

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE rzsm_lib GTest::gtest_main)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Agent and env tests need MuJoCo assets
set(MUJOCO_TEST_SOURCES test_mujoco_env.cpp test_agents.cpp)
foreach(test_src ${MUJOCO_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE rzsm_lib GTest::gtest_main)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
