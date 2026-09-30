cmake_minimum_required(VERSION 3.16)
project(LiteRT_LM_Multimodal_CLI VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 헤더 경로
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
)

# 라이브러리 경로
set(LITERT_LIB_DIR "${CMAKE_SOURCE_DIR}/lib")
link_directories(${LITERT_LIB_DIR})

# FTXUI 가져오기
include(FetchContent)
FetchContent_Declare(
    ftxui
    GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI
    GIT_TAG v6.1.9
)
FetchContent_MakeAvailable(ftxui)

add_executable(multimodal_cli 
    main.cpp
    utils.cpp
    config.cpp
    llm_engine.cpp
    tui_interface.cpp
    server_mode.cpp
)

set_target_properties(multimodal_cli PROPERTIES
    BUILD_RPATH "${LITERT_LIB_DIR};$ORIGIN/lib"
    INSTALL_RPATH "${LITERT_LIB_DIR};$ORIGIN/lib"
)

target_link_libraries(multimodal_cli
    PRIVATE
    litert-lm
    ftxui::screen
    ftxui::dom
    ftxui::component
    pthread
    dl
)
