load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_import", "cc_library")

cc_import(
    name = "litert_lm_import",
    shared_library = "lib/liblitert-lm.so",
)

cc_library(
    name = "litert_lm",
    hdrs = [
        "include/c/conversation.h",
        "include/c/engine.h",
    ],
    includes = ["include"],
    deps = [":litert_lm_import"],
)

cc_binary(
    name = "multimodal_cli",
    srcs = [
        "config.cpp",
        "config.h",
        "httplib.h",
        "json.hpp",
        "llm_engine.cpp",
        "llm_engine.h",
        "main.cpp",
        "server_mode.cpp",
        "server_mode.h",
        "tui_interface.cpp",
        "tui_interface.h",
        "utils.cpp",
        "utils.h",
    ],
    copts = [
        "-std=c++17",
        "-O3",
    ],
    linkopts = [
        "-lpthread",
        "-ldl",
        "-Wl,-rpath,$ORIGIN/lib",
        "-Wl,-rpath,/home/kenonix/gits/Ollama-LiteRT-LM-cpp/lib",
    ],
    deps = [
        ":litert_lm",
        "@ftxui//:component",
        "@ftxui//:dom",
        "@ftxui//:screen",
    ],
)
