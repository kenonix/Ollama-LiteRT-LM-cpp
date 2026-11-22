use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_llama_cpp)");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| manifest_dir.clone());

    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let arch_dir_name = match arch.as_str() {
        "x86_64" => Some("x86_64"),
        "aarch64" => Some("aarch64"),
        _ => None,
    };

    let lib_dir = repo_root.join("lib");
    let mut rpaths: Vec<String> = Vec::new();

    if let Some(arch_name) = arch_dir_name {
        let arch_lib_dir = lib_dir.join(arch_name);
        if arch_lib_dir.exists() {
            println!("cargo:rustc-link-search=native={}", arch_lib_dir.display());
            rpaths.push(arch_lib_dir.display().to_string());
        }
    }
    if lib_dir.exists() {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        rpaths.push(lib_dir.display().to_string());
    }

    println!("cargo:rustc-link-lib=dylib=litert-lm");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=pthread");

    // Optional llama.cpp GGUF + MTP support (via LLAMA_CPP_DIR or $HOME/llama.cpp)
    let llama_dir = env::var("LLAMA_CPP_DIR")
        .ok()
        .map(PathBuf::from)
        .or_else(|| env::var("HOME").ok().map(|h| PathBuf::from(h).join("llama.cpp")));

    if let Some(llama_dir) = llama_dir {
        let llama_h = llama_dir.join("include").join("llama.h");
        let llama_bin = llama_dir.join("build").join("bin");
        let libllama = llama_bin.join("libllama.so");

        if llama_h.exists() && libllama.exists() {
            println!("cargo:rustc-cfg=has_llama_cpp");
            println!("cargo:rerun-if-changed=csrc/llama_shim.cpp");
            println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");

            cc::Build::new()
                .cpp(true)
                .std("c++17")
                .file("csrc/llama_shim.cpp")
                .include(llama_dir.join("include"))
                .include(llama_dir.join("ggml").join("include"))
                .include(llama_dir.join("common"))
                .include(&repo_root)
                .warnings(false)
                .compile("llama_shim");

            println!("cargo:rustc-link-search=native={}", llama_bin.display());
            println!("cargo:rustc-link-lib=dylib=llama");
            println!("cargo:rustc-link-lib=dylib=llama-common");
            println!("cargo:rustc-link-lib=dylib=ggml");
            println!("cargo:rustc-link-lib=dylib=ggml-base");
            println!("cargo:rustc-link-lib=dylib=ggml-cpu");
            if llama_bin.join("libggml-cuda.so").exists() {
                println!("cargo:rustc-link-lib=dylib=ggml-cuda");
            }
            rpaths.push(llama_bin.display().to_string());
        }
    }

    for rp in rpaths {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", rp);
    }
}
