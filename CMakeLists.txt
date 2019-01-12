cmake_minimum_required(VERSION 2.8)
project( Exercise2 )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( Exercise2 Exercise2.cpp)
target_link_libraries( Exercise2 ${OpenCV_LIBS} )