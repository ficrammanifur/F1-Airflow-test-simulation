cmake_minimum_required(VERSION 3.10)
project(F1AirflowSimulator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm REQUIRED)

# Include directories
include_directories(${OPENGL_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})

# Create executable
add_executable(f1_airflow_simulator 
    f1_airflow_simulator.cpp
)

# Link libraries
target_link_libraries(f1_airflow_simulator 
    ${OPENGL_LIBRARIES}
    glfw
    GLEW::GLEW
    glm::glm
)

# For Windows
if(WIN32)
    target_link_libraries(f1_airflow_simulator 
        opengl32
        gdi32
        user32
        kernel32
    )
endif()

# For Linux
if(UNIX AND NOT APPLE)
    target_link_libraries(f1_airflow_simulator 
        GL
        GLU
        X11
        Xrandr
        Xinerama
        Xcursor
        pthread
        dl
    )
endif()

# Set output directory
set_target_properties(f1_airflow_simulator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create data directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/data)

# Copy any data files if they exist
if(EXISTS ${CMAKE_SOURCE_DIR}/data)
    file(COPY ${CMAKE_SOURCE_DIR}/data/ DESTINATION ${CMAKE_BINARY_DIR}/data/)
endif()

# Compiler-specific options
if(MSVC)
    target_compile_options(f1_airflow_simulator PRIVATE /W4)
else()
    target_compile_options(f1_airflow_simulator PRIVATE -Wall -Wextra -O3)
endif()

# Print build information
message(STATUS "OpenGL found: ${OPENGL_FOUND}")
message(STATUS "GLFW found: ${glfw3_FOUND}")
message(STATUS "GLEW found: ${GLEW_FOUND}")
message(STATUS "GLM found: ${glm_FOUND}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")