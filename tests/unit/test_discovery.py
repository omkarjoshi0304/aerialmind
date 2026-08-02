"""Tests for the hardware discovery service.

Verifies:
1. discover_hardware() returns a valid HardwareCapabilities instance
2. HardwareCapabilities is frozen (immutable)
3. Collection fields are tuples (not lists)
4. Probe results are consistent across calls
"""

from __future__ import annotations

import pytest

from aerialmind.hal.discovery import HardwareCapabilities, discover_hardware

# ---------------------------------------------------------------------------
# HardwareCapabilities dataclass
# ---------------------------------------------------------------------------


class TestHardwareCapabilities:
    def test_is_frozen(self) -> None:
        caps = discover_hardware()
        with pytest.raises(AttributeError):
            caps.has_cuda = True  # type: ignore[misc]

    def test_camera_devices_is_tuple(self) -> None:
        caps = discover_hardware()
        assert isinstance(caps.camera_devices, tuple)

    def test_serial_devices_is_tuple(self) -> None:
        caps = discover_hardware()
        assert isinstance(caps.serial_devices, tuple)


# ---------------------------------------------------------------------------
# discover_hardware() function
# ---------------------------------------------------------------------------


class TestDiscoverHardware:
    def test_returns_hardware_capabilities(self) -> None:
        caps = discover_hardware()
        assert isinstance(caps, HardwareCapabilities)

    def test_platform_hint_is_string(self) -> None:
        caps = discover_hardware()
        assert isinstance(caps.platform_hint, str)
        assert len(caps.platform_hint) > 0

    def test_has_onnxruntime_reflects_import(self) -> None:
        caps = discover_hardware()
        try:
            import onnxruntime  # noqa: F401

            expected = True
        except ImportError:
            expected = False
        assert caps.has_onnxruntime is expected

    def test_consistent_across_calls(self) -> None:
        caps1 = discover_hardware()
        caps2 = discover_hardware()
        assert caps1.has_cuda == caps2.has_cuda
        assert caps1.has_hailo == caps2.has_hailo
        assert caps1.has_onnxruntime == caps2.has_onnxruntime
        assert caps1.platform_hint == caps2.platform_hint
