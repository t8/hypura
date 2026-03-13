use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let llama_dir = PathBuf::from(&manifest_dir).join("../vendor/llama.cpp");
    let llama_dir = llama_dir.canonicalize().expect(
        "vendor/llama.cpp not found — run: git submodule update --init --recursive",
    );

    // Build llama.cpp via cmake
    let dst = cmake::Config::new(&llama_dir)
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("GGML_METAL", "ON")
        .define("GGML_METAL_EMBED_LIBRARY", "ON")
        .define("GGML_CPU", "ON")
        .define("GGML_BLAS", "OFF")
        .define("GGML_OPENMP", "OFF")
        .build();

    let lib_dir = dst.join("lib");

    // Link the static libraries produced by cmake
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-metal");

    // Link macOS frameworks
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=MetalKit");
    println!("cargo:rustc-link-lib=framework=Accelerate");

    // C++ standard library
    println!("cargo:rustc-link-lib=c++");

    // Generate Rust bindings via bindgen
    let include_llama = llama_dir.join("include");
    let include_ggml = llama_dir.join("ggml/include");

    let bindings = bindgen::Builder::default()
        .header(
            PathBuf::from(&manifest_dir)
                .join("wrapper.h")
                .to_str()
                .unwrap()
                .to_string(),
        )
        .clang_arg(format!("-I{}", include_llama.display()))
        .clang_arg(format!("-I{}", include_ggml.display()))
        .allowlist_function("llama_.*")
        .allowlist_function("ggml_.*")
        .allowlist_function("gguf_.*")
        .allowlist_type("llama_.*")
        .allowlist_type("ggml_.*")
        .allowlist_type("gguf_.*")
        .allowlist_var("LLAMA_.*")
        .allowlist_var("GGML_.*")
        .allowlist_var("GGUF_.*")
        .derive_debug(true)
        .derive_default(true)
        .generate()
        .expect("Failed to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Failed to write bindings");

    // Rebuild if llama.cpp sources change
    println!("cargo:rerun-if-changed=wrapper.h");
    println!(
        "cargo:rerun-if-changed={}",
        llama_dir.join("include").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        llama_dir.join("src").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        llama_dir.join("ggml").display()
    );
}
