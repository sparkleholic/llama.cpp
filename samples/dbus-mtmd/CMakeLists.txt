cmake_minimum_required(VERSION 3.16)
project(dbus-mtmd)

set(CMAKE_CXX_STANDARD 17)

# find_package(ggml REQUIRED CONFIG PATHS /usr/local/lib/cmake/ggml)
find_package(llama REQUIRED CONFIG PATHS /usr/local/lib/cmake/llama)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SDBUSCPP REQUIRED sdbus-c++)

include_directories(
    /usr/local/include
    ${SDBUSCPP_INCLUDE_DIRS}
)

link_directories(
    /usr/local/lib
    ${SDBUSCPP_LIBRARY_DIRS}
)

add_executable(dbus-mtmd main.cpp llme_service.cpp)
target_link_libraries(dbus-mtmd
    llama
    mtmd
    ${SDBUSCPP_LIBRARIES}
)

# Test executable for queryImage
add_executable(test_queryimage test_queryimage.cpp llme_service.cpp)
target_link_libraries(test_queryimage
    llama
    mtmd
    ${SDBUSCPP_LIBRARIES}
)
