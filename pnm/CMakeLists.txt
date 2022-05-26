cmake_minimum_required(VERSION 3.17)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(pnm)

find_package(pybind11 REQUIRED)

find_package(Eigen3 REQUIRED)

add_dependencies(netgrid netgrid)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../netgrid_old)

set(SOURCE_CODE Pnm.cpp)

add_library(${PROJECT_NAME} ${SOURCE_CODE})

target_link_libraries(${PROJECT_NAME} PUBLIC
        Eigen3::Eigen
        netgrid)

pybind11_add_module(${PROJECT_NAME}_bind wrapper.cpp)

target_link_libraries(${PROJECT_NAME}_bind PRIVATE ${PROJECT_NAME})

set_target_properties(${PROJECT_NAME}_bind PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../..)