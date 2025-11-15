# Linear Programming tests.
add_executable(test_simplex test_simplex.cpp)

# Link GTest and pthread
target_link_libraries(test_simplex
  ${PROJECT_NAME}
  ${GTEST_LIBRARIES}
  pthread
)

# Make the test runnable by ctest. I am not using the more generic command
# gtest_discover_tests() because for some reason it does not pass the argument
# to the executable, thus throwing an exception.
gtest_add_tests(TARGET test_simplex EXTRA_ARGS "${CMAKE_CURRENT_LIST_DIR}/problems")


# Quadratic Programming tests.
add_executable(test_qp test_qp.cpp)

# Link GTest and pthread
target_link_libraries(test_qp
  ${PROJECT_NAME}
  ${GTEST_LIBRARIES}
  pthread
)

target_compile_definitions(test_qp PRIVATE EigenOpt_QUADPROG_DEBUG_ON EigenOpt_QUADPROG_SILENCE_DBG)

# Make the test runnable by ctest. I am not using the more generic command
# gtest_discover_tests() because for some reason it does not pass the argument
# to the executable, thus throwing an exception.
gtest_add_tests(TARGET test_qp EXTRA_ARGS "${CMAKE_CURRENT_LIST_DIR}/problems")
