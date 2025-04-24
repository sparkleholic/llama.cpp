cmake_minimum_required(VERSION 3.10)
project(webcam_capture)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add executable
add_executable(webcam_capture webcam_capture.cpp)

# Link OpenCV libraries
target_link_libraries(webcam_capture PRIVATE ${OpenCV_LIBS}) 