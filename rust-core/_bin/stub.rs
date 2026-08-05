use pyo3_stub_gen::Result;
use std::path::PathBuf;

fn main() -> Result<()> {
    let stub = maxsimd::stub_info()?;
    stub.generate()?;

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let init_py = manifest.join("bbq/maxsimd/__init__.py");

    if let Some(parent) = init_py.parent() {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::write(&init_py, "from .maxsimd import *\n")?;
    println!("Generated `maxsimd` stub and __init__.py");

    Ok(())
}
