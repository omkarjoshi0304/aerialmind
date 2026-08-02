"""Hardware discovery service.

Probes the system at startup to detect available accelerators, cameras,
and serial devices. Every probe is wrapped in try/except so the service
returns safe defaults on any platform — including macOS dev machines
where no drone hardware exists.
"""

from __future__ import annotations

import glob
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HardwareCapabilities:
    """Snapshot of detected hardware on the current system."""

    has_cuda: bool
    cuda_device_name: str | None
    has_hailo: bool
    has_onnxruntime: bool
    camera_devices: tuple[str, ...]
    serial_devices: tuple[str, ...]
    platform_hint: str


def _probe_cuda() -> tuple[bool, str | None]:
    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as drv

        return True, drv.Device(0).name()
    except Exception:
        return False, None


def _probe_hailo() -> bool:
    try:
        import hailo_platform  # noqa: F401

        return True
    except Exception:
        return False


def _probe_onnxruntime() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except Exception:
        return False


def _probe_camera_devices() -> tuple[str, ...]:
    if sys.platform != "linux":
        return ()
    return tuple(sorted(glob.glob("/dev/video*")))


def _probe_serial_devices() -> tuple[str, ...]:
    if sys.platform != "linux":
        return ()
    devices: list[str] = []
    for pattern in ("/dev/ttyTHS*", "/dev/ttyAMA*", "/dev/ttyUSB*"):
        devices.extend(glob.glob(pattern))
    return tuple(sorted(devices))


def _detect_platform_hint(
    has_cuda: bool,
    has_hailo: bool,
    camera_devices: tuple[str, ...],
    serial_devices: tuple[str, ...],
) -> str:
    if Path("/etc/nv_tegra_release").exists():
        return "jetson"
    try:
        model = Path("/proc/device-tree/model").read_text()
        if "Raspberry Pi" in model:
            return "rpi"
    except (FileNotFoundError, PermissionError):
        pass

    if has_cuda:
        return "jetson"
    if has_hailo:
        return "rpi"

    if camera_devices or serial_devices:
        return "unknown"

    return "sim"


def discover_hardware() -> HardwareCapabilities:
    """Probe the system and return a snapshot of available hardware."""
    has_cuda, cuda_device_name = _probe_cuda()
    has_hailo = _probe_hailo()
    has_onnxruntime = _probe_onnxruntime()
    camera_devices = _probe_camera_devices()
    serial_devices = _probe_serial_devices()
    platform_hint = _detect_platform_hint(
        has_cuda, has_hailo, camera_devices, serial_devices,
    )

    return HardwareCapabilities(
        has_cuda=has_cuda,
        cuda_device_name=cuda_device_name,
        has_hailo=has_hailo,
        has_onnxruntime=has_onnxruntime,
        camera_devices=camera_devices,
        serial_devices=serial_devices,
        platform_hint=platform_hint,
    )
