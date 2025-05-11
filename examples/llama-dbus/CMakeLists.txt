set(TARGET llama-dbus)
add_executable(${TARGET} llama-dbus.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} dbus-1)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Find DBus package
find_package(PkgConfig REQUIRED)
pkg_check_modules(DBUS REQUIRED dbus-1)
target_include_directories(${TARGET} PRIVATE ${DBUS_INCLUDE_DIRS})
target_link_directories(${TARGET} PRIVATE ${DBUS_LIBRARY_DIRS})

# Add client example
set(TARGET_CLIENT llama-dbus-client)
add_executable(${TARGET_CLIENT} llama-dbus-client.cpp)
install(TARGETS ${TARGET_CLIENT} RUNTIME)
target_link_libraries(${TARGET_CLIENT} PRIVATE dbus-1)
target_compile_features(${TARGET_CLIENT} PRIVATE cxx_std_17)
target_include_directories(${TARGET_CLIENT} PRIVATE ${DBUS_INCLUDE_DIRS})
target_link_directories(${TARGET_CLIENT} PRIVATE ${DBUS_LIBRARY_DIRS}) 