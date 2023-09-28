project(TestLib)

cmake_minimum_required(VERSION 3.5)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(Eigen3::Eigen3 ALIAS Eigen3::Eigen)
option(BUILD_TESTS "" OFF)
add_subdirectory(libfranka EXCLUDE_FROM_ALL)

find_package(JlCxx)
get_target_property(JlCxx_location JlCxx::cxxwrap_julia LOCATION)
get_filename_component(JlCxx_location ${JlCxx_location} DIRECTORY)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${JlCxx_location}")
message(STATUS "Found JlCxx at ${JlCxx_location}")

add_library(frankajl SHARED frankajl.cc)
target_link_libraries(frankajl PRIVATE
  JlCxx::cxxwrap_julia
  Franka::Franka
  Eigen3::Eigen
)
target_compile_options(frankajl PRIVATE -Wall -Wextra -pedantic -Werror)
target_compile_features(frankajl PRIVATE cxx_std_17)

install(TARGETS
  frankajl
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib
)
