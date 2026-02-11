cmake_minimum_required(VERSION 3.12.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(lab02 LANGUAGES CXX C)

include(../../common/cmake-common.cmake)

add_executable(exercise-01 src/exercise-01.cpp src/Poisson2D.cpp)
deal_ii_setup_target(exercise-01)
