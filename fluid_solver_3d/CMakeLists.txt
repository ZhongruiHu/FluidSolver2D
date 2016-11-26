cmake_minimum_required (VERSION 2.8)
project (Fluid_Solver)

file(GLOB SOURCES
    "*.h"
	"*.hpp"
    "*.cpp"
)
set(EXECUTABLE_NAME "fluid_solver3d")
# add the executable
add_executable(${EXECUTABLE_NAME} ${SOURCES})

# HANDLE OPENCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

# Detect and add SFML
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})
#Find any version 2.X of SFML
#See the FindSFML.cmake file for additional details and instructions
find_package(SFML 2 REQUIRED system window graphics)
if(SFML_FOUND)
  include_directories(${SFML_INCLUDE_DIR})
  target_link_libraries(${EXECUTABLE_NAME} ${SFML_LIBRARIES})
endif()
 


target_include_directories (${EXECUTABLE_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (${EXECUTABLE_NAME} ${OpenCL_LIBRARY})
