"""
Unit tests for hardware and CPU architecture compatibility checks.
"""

from unittest.mock import mock_open, patch
import pytest

from bbq.src.common.hardware import check_cpu_support, verify_hardware_or_exit


def test_check_cpu_support_current_machine():
    """Verify that the current development/test machine passes the check."""
    is_supported, reason = check_cpu_support()
    assert is_supported is True
    assert reason == ""


@patch("platform.machine", return_value="aarch64")
def test_check_cpu_support_unsupported_arch(mock_arch):
    """Verify that non-x86_64 architectures (e.g., aarch64, arm64) are rejected."""
    is_supported, reason = check_cpu_support()
    assert is_supported is False
    assert "Unsupported CPU architecture: 'aarch64'" in reason
    assert "x86_64" in reason


@patch("platform.machine", return_value="x86_64")
@patch("sys.platform", "linux")
def test_check_cpu_support_missing_fma_on_linux(mock_arch):
    """Verify that an x86_64 CPU without the FMA flag on Linux is rejected."""
    cpuinfo_without_fma = "flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov avx2\n"
    with patch("builtins.open", mock_open(read_data=cpuinfo_without_fma)):
        is_supported, reason = check_cpu_support()
        assert is_supported is False
        assert "FMA" in reason


@patch(
    "bbq.src.common.hardware.check_cpu_support",
    return_value=(False, "Mocked unsupported CPU"),
)
def test_verify_hardware_or_exit_terminates(mock_check, capsys):
    """Verify that verify_hardware_or_exit exits with code 1 and writes to stderr on unsupported hardware."""
    with pytest.raises(SystemExit) as exc_info:
        verify_hardware_or_exit(exit_code=42)

    assert exc_info.value.code == 42
    err = capsys.readouterr().err
    assert "ERROR: BBQ-RAG Hardware Compatibility Check Failed" in err
    assert "Mocked unsupported CPU" in err


@patch("bbq.src.common.hardware.check_cpu_support", return_value=(True, ""))
def test_verify_hardware_or_exit_passes(mock_check):
    """Verify that verify_hardware_or_exit does not exit when hardware is supported."""
    # Should complete without raising SystemExit
    verify_hardware_or_exit()
