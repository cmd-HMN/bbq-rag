use pyo3_stub_gen::Result;
use std::path::PathBuf;
use std::process::Command;

fn main() -> Result<()> {
    let stub = maxsimd::stub_info()?;
    stub.generate()?;

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let init_py = manifest.join("bbq/maxsimd/__init__.py");
    let init_pyi = manifest.join("bbq/maxsimd/__init__.pyi");

    if let Some(parent) = init_py.parent() {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::write(&init_py, "from .maxsimd import *\n")?;
    println!("Generated `maxsimd` stub and __init__.py");
    
    // add version to it
    if init_pyi.exists(){
        let mut pyi_content = std::fs::read_to_string(&init_pyi)?;
        if !pyi_content.contains("__version__") {
            pyi_content = format!("{}__version__ = '{}'\n", pyi_content, env!("CARGO_PKG_VERSION"));
            std::fs::write(&init_pyi, pyi_content)?;
        }
    }

    // fomrat 
    let _ = Command::new("ruff")
        .args(["format", init_py.to_str().unwrap(), init_pyi.to_str().unwrap()]).status();

    Ok(())
}
