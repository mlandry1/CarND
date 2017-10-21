project(GNB-CPP)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(sources
    ./main.cpp
    ./classifier.cpp
    )

add_executable(GNB-CPP ${sources})
