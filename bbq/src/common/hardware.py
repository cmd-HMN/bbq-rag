from __future__ import annotations

import platform
import sys
from typing import Tuple


def check_cpu_support() -> Tuple[bool, str]:
    """
    Checks whether the host machine supports the necessary architecture
    (x86_64) and CPU SIMD instructions (FMA).

    Returns:
        (is_supported, error_message): Tuple where is_supported is True if compatible,
        or False with a descriptive error message explaining why the machine is unsupported.
    """
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64", "x64"):
        return (
            False,
            f"Unsupported CPU architecture: '{machine}'. "
            f"BBQ-RAG is only supported on x86_64 systems with FMA instruction support.",
        )

    system = sys.platform.lower()
    has_fma = False

    if system.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("flags") or line.startswith("Features"):
                        flags = set(line.split(":", 1)[1].strip().lower().split())
                        if "fma" in flags:
                            has_fma = True
                            break
        except Exception:
            # Fallback if /proc/cpuinfo cannot be read
            has_fma = True
    elif system == "darwin":
        import subprocess

        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.features", "machdep.cpu.leaf7_features"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            if "FMA" in out.upper():
                has_fma = True
        except Exception:
            has_fma = True
    elif system.startswith("win"):
        has_fma = True
    else:
        has_fma = True

    if not has_fma:
        return (
            False,
            "Your x86_64 processor does not support FMA (Fused Multiply-Add) instructions. "
            "BBQ-RAG requires FMA-capable processors (Intel Haswell+, AMD Piledriver+).",
        )

    return True, ""


def verify_hardware_or_exit(exit_code: int = 1) -> None:
    """
    Verifies that the current machine meets CPU and instruction set requirements.
    If incompatible, prints a clear error message to stderr and terminates.
    """
    is_supported, reason = check_cpu_support()
    if not is_supported:
        sys.stderr.write(
            f"ERROR: BBQ-RAG Hardware Compatibility Check Failed\n"
            f"{reason}\n"
            f"BBQ-RAG cannot be installed or run on this machine.\n"
        )
        sys.stderr.flush()
        sys.exit(exit_code)
