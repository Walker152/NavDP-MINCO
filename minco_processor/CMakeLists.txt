cmake_minimum_required(VERSION 3.16)
project(minco_processor VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(Eigen3 REQUIRED)
find_package(OpenMP)
include(CMakePackageConfigHelpers)

file(GLOB UTILS_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cpp"
)

set(TRAJ_OPT_SOURCES
  src/traj_opt/minco_optimizer.cpp
  src/traj_opt/yaw_traj_opt.cpp
)

set(MINCO_ALGORITHM_SOURCES
  src/minco_processor/minco_pipeline.cpp
  src/minco_core/components/trajectory_safety_checker.cpp
  src/static_sim_esdf_map.cpp
)

add_library(${PROJECT_NAME}
  ${UTILS_SOURCES}
  ${TRAJ_OPT_SOURCES}
  ${MINCO_ALGORITHM_SOURCES}
)

target_compile_definitions(${PROJECT_NAME}
PUBLIC
  ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/"
)

target_include_directories(${PROJECT_NAME}
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME}
PUBLIC
  Eigen3::Eigen
)

set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(minco_processor_compile_test
  tests/test_pure_algorithm_compile.cpp
)
target_link_libraries(minco_processor_compile_test PRIVATE ${PROJECT_NAME})

find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
  pybind11_add_module(minco_processor_py
    bindings/minco_pybind.cpp
  )
  target_link_libraries(minco_processor_py PRIVATE ${PROJECT_NAME})
  set_target_properties(minco_processor_py PROPERTIES OUTPUT_NAME _minco_processor)
endif()

install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(FILES
  include/data_structure/base/trajectory.h
  include/data_structure/base/ellipsoid.h
  include/data_structure/base/piece.h
  DESTINATION include/data_structure/base
)
install(FILES include/minco_core/components/trajectory_safety_checker.hpp
  DESTINATION include/minco_core/components
)
install(DIRECTORY include/minco_processor
  DESTINATION include
)
install(DIRECTORY include/traj_opt
  DESTINATION include
)
install(DIRECTORY include/utils
  DESTINATION include
  PATTERN "header/matplotlibcpp.hpp" EXCLUDE
  PATTERN "header/tinycolormap.hpp" EXCLUDE
  PATTERN "header/yaml_loader.hpp" EXCLUDE
)

install(EXPORT ${PROJECT_NAME}Targets
  NAMESPACE minco_processor::
  DESTINATION lib/cmake/${PROJECT_NAME}
)

configure_package_config_file(
  cmake/minco_processorConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/minco_processorConfig.cmake"
  INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/minco_processorConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/minco_processorConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/minco_processorConfigVersion.cmake"
  DESTINATION lib/cmake/${PROJECT_NAME}
)
