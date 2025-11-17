### This file is an adaptation derived from
###   github.com/francofusco/template-cmake-project/blob/main/CMakeLists.txt
### Original Author: Franco FUSCO
### Source: github.com/francofusco/template-cmake-project
cmake_minimum_required(VERSION 3.19.4)
project(EigenOpt VERSION 1.0.0)

# Allow using colors in messages. Use with parsimony ;)
include(cmake/colors.cmake)

#######################
# LOCATE DEPENDENCIES #
#######################

find_package(Eigen3 REQUIRED)


#############
# LIBRARIES #
#############

# Add a "header-only-library".
add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME}
  INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(${PROJECT_NAME} INTERFACE Eigen3::Eigen)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  # target_compile_options(${PROJECT_NAME} INTERFACE -O0)
  # target_compile_options(${PROJECT_NAME} INTERFACE -Wall)
  # target_compile_definitions(${PROJECT_NAME} INTERFACE ${PROJECT_NAME}_SIMPLEX_DEBUG_ON)
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  # target_compile_options(${PROJECT_NAME} INTERFACE -O2)
endif()

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)


############
# BINARIES #
############

# Compile one simple example per solver.
foreach(EXAMPLE IN ITEMS simplex_example qp_example)
  add_executable(${EXAMPLE} src/${EXAMPLE}.cpp)
  target_link_libraries(${EXAMPLE} PRIVATE ${PROJECT_NAME})

  # Remove issues related to the size of the compiled code.
  if(WIN32)
    target_compile_options(${EXAMPLE} PRIVATE -O2 -g0 -flto)
    target_link_options(${EXAMPLE} PRIVATE -flto)
  endif()
endforeach()

# If piksel has been downloaded, compile the robot example.
set(PIKSEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/piksel")
if(EXISTS "${PIKSEL_DIR}" AND IS_DIRECTORY "${PIKSEL_DIR}")
  find_package(VISP REQUIRED)
  message(STATUS "Found ${Green}piksel${ColourReset} directory - will compile ${Green}example_pixel_robot.cpp${ColourReset}")
  add_subdirectory(piksel)
  add_executable(example_pixel_robot src/example_pixel_robot.cpp)
  target_include_directories(example_pixel_robot PRIVATE piksel/src ${VISP_INCLUDE_DIRS})
  target_link_libraries(example_pixel_robot PRIVATE piksel ${PROJECT_NAME} ${VISP_LIBRARIES})
endif()

###########
# INSTALL #
###########

include(GNUInstallDirs)

# Installing a library is a little more involved :P
install(
  TARGETS ${PROJECT_NAME}
  DESTINATION lib
  EXPORT ${PROJECT_NAME}Targets
)

# Copy all headers from the "include" folder
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}
  DESTINATION include
)

# Install the "Targets" file as well.
install(
  EXPORT ${PROJECT_NAME}Targets
  NAMESPACE ${PROJECT_NAME}::
  FILE ${PROJECT_NAME}Targets.cmake
  DESTINATION lib/cmake/${PROJECT_NAME}
)

# Helpers to create the "package config" files.
include(CMakePackageConfigHelpers)

# Create the "Config.cmake" file (to allow "find_package"-ing this project).
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION "lib/cmake/example"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# Generate the version file associated to the "Config.cmake" file.
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  VERSION "${${PROJECT_NAME}_VERSION_MAJOR}.${${PROJECT_NAME}_VERSION_MINOR}"
  COMPATIBILITY AnyNewerVersion
)

# Install the configuration file.
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION lib/cmake/${PROJECT_NAME}
)

# Make the project usable from the build directory as well.
export(
  EXPORT ${PROJECT_NAME}Targets
  NAMESPACE ${PROJECT_NAME}::
  FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake"
)

# add the possibility to run 'make uninstall_EigenOpt' to remove files added via 'make install'
# NOTE: this will remove files only, and not their parent directories.
# WARNING: this will work only if you do not play around the installed targets manually,
# and if you do not change the install destination.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/uninstall.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/uninstall_${PROJECT_NAME}.cmake
  IMMEDIATE @ONLY
)
add_custom_target(uninstall_${PROJECT_NAME} COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/uninstall_${PROJECT_NAME}.cmake")


###########
# TESTING #
###########
find_package(GTest QUIET)
option(BUILD_TESTS "Enable building unit tests" ${GTEST_FOUND})

if(${BUILD_TESTS})
  if(${GTEST_FOUND})
    message(STATUS "Tests will be built using GTest.")
    enable_testing()
    add_subdirectory(test)
  else(${GTEST_FOUND})
    message(FATAL_ERROR "BUILD_TESTS is ON, but GTest could not be found. Tests will not be compiled!")
  endif(${GTEST_FOUND})
else(${BUILD_TESTS})
  message(STATUS "Tests will NOT be run. If you have GTest installed, you can enable them via -DBUILD_TESTS=ON.")
endif(${BUILD_TESTS})

#########################
# DOXYGEN DOCUMENTATION #
#########################

add_subdirectory(doc)
