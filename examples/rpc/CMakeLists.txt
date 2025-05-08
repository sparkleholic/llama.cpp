set(TARGET rpc-server)
add_executable(${TARGET} rpc-server.cpp)

find_package(PkgConfig REQUIRED)
pkg_check_modules(AVAHI REQUIRED avahi-client)

message(STATUS "AVAHI_INCLUDE_DIRS: ${AVAHI_INCLUDE_DIRS}")
message(STATUS "AVAHI_LIBRARY_DIRS: ${AVAHI_LIBRARY_DIRS}")
message(STATUS "AVAHI_LIBRARIES: ${AVAHI_LIBRARIES}")

include_directories(
    ${AVAHI_INCLUDE_DIRS}
)
link_directories(
    ${AVAHI_LIBRARY_DIRS}
)

target_link_libraries(${TARGET} 
    PRIVATE 
    ggml 
    ${AVAHI_LIBRARIES}
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
