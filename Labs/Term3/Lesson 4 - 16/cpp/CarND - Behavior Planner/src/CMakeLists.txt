project(BehaviorPlanner-CPP)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(sources
    ./main.cpp
    ./road.cpp
    ./vehicle.cpp
    )

add_executable(BehaviorPlanner-CPP ${sources})
