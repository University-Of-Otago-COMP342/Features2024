CMAKE_MINIMUM_REQUIRED(VERSION "3.12")

set(CMAKE_CXX_STANDARD 14)
#this is set manually for the lab computers this should not be required for other systems with proper installation
#DELETE THE FOLLOWING LINE IF YOU WORK ON ANY OTHER MACHINES THAN THE LAB COMPUTERS, THIS POINTS TO K:
#set(CMAKE_TOOLCHAIN_FILE "K:\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake")
project(Features)

#set (OPENCV_DIR "K:\\vcpkg\\installed\\x64-windows\\share\\opencv" )
find_package (OpenCV CONFIG REQUIRED)
include_directories (${OpenCV_INCLUDE_DIRS})

#link_directories("/usr/local/opt/opencv/lib/")
add_executable(features features.cpp Timer.h Timer.cpp  )
target_link_libraries(features
  ${OpenCV_LIBS}
)
