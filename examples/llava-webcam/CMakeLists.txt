cmake_minimum_required(VERSION 3.10)
project(webcam_llava)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Find Threads package
find_package(Threads REQUIRED)

# Set the path to llama.cpp root directory (assuming this example is two levels deep)
set(LLAMA_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# Add necessary include directories
include_directories(
    ${LLAMA_CPP_DIR} 
    ${LLAMA_CPP_DIR}/common
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/examples 
    ${LLAMA_CPP_DIR}/examples/llava
    ${OpenCV_INCLUDE_DIRS} # Add OpenCV includes explicitly
)

# Add executable
add_executable(webcam_llava webcam_llava.cpp)

# Link against system-installed llama, mtmd, common and OpenCV
# target_link_libraries(webcam_llava PRIVATE 
#     ${OpenCV_LIBS}
#     /usr/local/lib/libllama.so
#     /usr/local/lib/libmtmd_shared.so  # Assuming this exists
#     Threads::Threads # Link pthreads explicitly
# )
target_link_libraries(webcam_llava PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} llava /usr/local/lib/libmtmd_shared.so)

# Ensure OpenCV libraries are found and linked
if(OpenCV_FOUND)
  target_link_libraries(webcam_llava PRIVATE ${OpenCV_LIBS})
else()
  message(FATAL_ERROR "OpenCV not found!")
endif() 