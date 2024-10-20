cmake_minimum_required(VERSION 3.22)

project(mls_mpm)

add_library(${PROJECT_NAME} SHARED
    src/main.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC ./include)

# set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ./Release)
