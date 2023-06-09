cmake_minimum_required(VERSION 3.20)

if(WIN32)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif ()

set(PROJECT convert)
project(${PROJECT})

##################
##### target #####
##################
# Target for this build
add_library(${PROJECT} SHARED)

target_sources(
    ${PROJECT}
    PRIVATE
    src/convert.cpp
    src/readGoZAndTriangulate.cpp
)
########################
##### target (end) #####
########################


#####################
##### Libraries #####
#####################
###
# json
find_package(nlohmann_json CONFIG REQUIRED)

###
# eigen3
find_package(Eigen3 CONFIG REQUIRED)

# Add external libraries
target_link_libraries(
    ${PROJECT}
    PRIVATE
    nlohmann_json
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
)
###########################
##### Libraries (end) #####
###########################


###################
##### Include #####
###################
# Add a include files
target_include_directories(
    ${PROJECT}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../submodule/libigl/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../submodule/eigen
    ${CMAKE_CURRENT_SOURCE_DIR}/../submodule/FromZ/src
)
#########################
##### Include (end) #####
#########################


###########################
##### compile options #####
###########################
# because we want to use std::filesystem, we require C++17
target_compile_features(${PROJECT} PRIVATE cxx_std_17)

if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4819")
    add_definitions(-D_WIN32_WINNT=0x0A00)
endif ()
#################################
##### compile options (end) #####
#################################
