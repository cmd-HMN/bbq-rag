import tomllib
from pathlib import Path


def _read_project_version() -> str:
    bases = [
        Path.cwd(),
        Path(__file__).resolve().parents[2],
        Path(__file__).resolve().parents[3],
    ]
    for base in bases:
        pyproject = base / "pyproject.toml"
        if pyproject.exists():
            try:
                with open(pyproject, "rb") as f:
                    v = tomllib.load(f).get("project", {}).get("version")
                    if v:
                        return str(v)
            except Exception:
                pass
        cargo = base / "Cargo.toml"
        if cargo.exists():
            try:
                with open(cargo, "rb") as f:
                    cv = tomllib.load(f).get("package", {}).get("version")
                    if cv:
                        return str(cv)
            except Exception:
                pass
    try:
        from importlib.metadata import version

        return version("bbq")
    except Exception:
        return "0.1.3-alpha"


# Global package version directly read from pyproject.toml / Cargo.toml
__version__ = _read_project_version()

__all__ = ["_read_project_version", "__version__"]
