cmake_minimum_required(VERSION 3.17)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# set(CMAKE_BUILD_TYPE Release)

set(POJECT_NAME vofpnm)
project(${POJECT_NAME})
add_subdirectory(../netgrid_old ${PROJECT_BINARY_DIR}/netgrid)
add_subdirectory(pnm)
add_subdirectory(vof)



