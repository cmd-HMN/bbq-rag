fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");

    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch != "x86_64" {
        eprintln!("ERROR: BBQ-RAG (maxsimd) is ONLY supported on x86_64 architecture with FMA.");
        eprintln!("Detected target architecture: '{}'.", target_arch);
        eprintln!("BBQ-RAG cannot be installed or run on this machine.");
        std::process::exit(1);
    }

    #[cfg(target_arch = "x86_64")]
    {
        let target = std::env::var("TARGET").unwrap_or_default();
        let host = std::env::var("HOST").unwrap_or_default();
        if target == host || target.is_empty() {
            if is_x86_feature_detected!("fma") {
            } else {
                eprintln!(
                    "ERROR: BBQ-RAG (maxsimd) requires FMA (Fused Multiply-Add) CPU instruction support."
                );
                eprintln!("Your CPU does not support FMA instructions.");
                eprintln!("BBQ-RAG cannot be installed or run on this machine.");
                std::process::exit(1);
            }
        }
    }
}
