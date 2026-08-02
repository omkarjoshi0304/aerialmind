"""HAL provider factory.

Reads a HALConfig and instantiates the correct provider classes
for the target platform. Provider registries map config strings
(e.g., "sim", "cpu") to concrete provider classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aerialmind.core.config import HALConfig, load_hal_config
from aerialmind.core.protocols import GPSHAL, IMUHAL, AcceleratorHAL, CameraHAL
from aerialmind.hal.accelerator.cpu_provider import CPUFallbackProvider
from aerialmind.hal.camera.sim_provider import SimCameraProvider
from aerialmind.hal.gps.sim_provider import SimGPSProvider
from aerialmind.hal.imu.sim_provider import SimIMUProvider

_CAMERA_PROVIDERS: dict[str, type[Any]] = {
    "sim": SimCameraProvider,
}

_ACCELERATOR_PROVIDERS: dict[str, type[Any]] = {
    "cpu": CPUFallbackProvider,
}

_IMU_PROVIDERS: dict[str, type[Any]] = {
    "sim": SimIMUProvider,
}

_GPS_PROVIDERS: dict[str, type[Any]] = {
    "sim": SimGPSProvider,
}


@dataclass(frozen=True)
class HALProviders:
    """Container for instantiated HAL provider objects."""

    camera: CameraHAL
    accelerator: AcceleratorHAL
    imu: IMUHAL | None
    gps: GPSHAL | None


def _lookup_provider(
    registry: dict[str, type[Any]],
    provider_name: str,
    kind: str,
) -> type[Any]:
    if provider_name not in registry:
        available = ", ".join(sorted(registry))
        msg = f"Unknown {kind} provider '{provider_name}'. Available: {available}"
        raise ValueError(msg)
    return registry[provider_name]


def create_hal_providers(config: HALConfig) -> HALProviders:
    """Instantiate HAL providers from a HALConfig.

    Camera and sensor providers are opened automatically.
    The accelerator is stateless until load_model() is called.
    """
    camera_cls = _lookup_provider(
        _CAMERA_PROVIDERS, config.camera.provider, "camera",
    )
    camera = camera_cls()
    camera.open(config.camera.model_dump())

    accel_cls = _lookup_provider(
        _ACCELERATOR_PROVIDERS, config.accelerator.provider, "accelerator",
    )
    accelerator = accel_cls()

    imu = None
    if config.imu is not None:
        imu_cls = _lookup_provider(
            _IMU_PROVIDERS, config.imu.provider, "IMU",
        )
        imu = imu_cls()
        imu.open(config.imu.model_dump())

    gps = None
    if config.gps is not None:
        gps_cls = _lookup_provider(
            _GPS_PROVIDERS, config.gps.provider, "GPS",
        )
        gps = gps_cls()
        gps.open(config.gps.model_dump())

    return HALProviders(
        camera=camera,
        accelerator=accelerator,
        imu=imu,
        gps=gps,
    )


def load_hal_from_yaml(path: str | Path) -> HALProviders:
    """Load a HAL config YAML and build providers in one step."""
    config = load_hal_config(path)
    return create_hal_providers(config)
