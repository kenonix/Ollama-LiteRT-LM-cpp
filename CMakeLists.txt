cmake_minimum_required(VERSION 3.16)
project(LiteRT_LM_Multimodal_CLI VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 시스템 아키텍처 자동 감지 (Jetson: aarch64, PC: x86_64)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ARCH_NAME "aarch64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(ARCH_NAME "x86_64")
else()
    set(ARCH_NAME "")
endif()

# 헤더 경로
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
)

# 라이브러리 경로
set(LITERT_LIB_DIR "${CMAKE_SOURCE_DIR}/lib")
if(ARCH_NAME AND EXISTS "${LITERT_LIB_DIR}/${ARCH_NAME}")
    link_directories("${LITERT_LIB_DIR}/${ARCH_NAME}" ${LITERT_LIB_DIR})
    set(ARCH_RPATH "${LITERT_LIB_DIR}/${ARCH_NAME};$ORIGIN/lib/${ARCH_NAME}")
else()
    link_directories(${LITERT_LIB_DIR})
    set(ARCH_RPATH "")
endif()

add_executable(multimodal_cli 
    main.cpp
    utils.cpp
    config.cpp
    llm_engine.cpp
    server_mode.cpp
)

set_target_properties(multimodal_cli PROPERTIES
    BUILD_RPATH "${ARCH_RPATH};${LITERT_LIB_DIR};$ORIGIN/lib"
    INSTALL_RPATH "${ARCH_RPATH};${LITERT_LIB_DIR};$ORIGIN/lib"
)

target_link_libraries(multimodal_cli
    PRIVATE
    litert-lm
    pthread
    dl
)
