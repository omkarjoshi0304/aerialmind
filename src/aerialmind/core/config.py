"""AerialMind configuration models and YAML loader.

Pydantic v2 models for system-level and HAL-level configuration.
All models are frozen (immutable after construction), matching the
project's frozen-dataclass convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from aerialmind.core.types import OperatingMode

# ---------------------------------------------------------------------------
# HAL config models
# ---------------------------------------------------------------------------


class CameraConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    width: int = 1280
    height: int = 720
    fps: int = 30
    format: str = "NV12"


class AcceleratorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    precision: str = "fp16"
    max_batch_size: int = 1
    workspace_mb: int = 512
    model_cache_dir: str = "/opt/aerialmind/model_cache"


class IMUConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    device: str = "/dev/ttyTHS1"
    baudrate: int = 921600
    sample_rate_hz: int = 400


class GPSConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    device: str = "/dev/ttyTHS0"
    baudrate: int = 115200


class HALConfig(BaseModel):
    """Hardware Abstraction Layer configuration for a specific platform."""

    model_config = ConfigDict(frozen=True)

    platform: str
    camera: CameraConfig
    accelerator: AcceleratorConfig
    imu: IMUConfig | None = None
    gps: GPSConfig | None = None


# ---------------------------------------------------------------------------
# System config models
# ---------------------------------------------------------------------------


class SafeZone(BaseModel):
    model_config = ConfigDict(frozen=True)

    lat: float
    lon: float
    radius_m: float
    label: str


class PathsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    hal_config: str = "config/hal/sim.yaml"
    roe_policy: str = "config/roe/civil_standard.yaml"
    mode_config: str = "config/modes/default.yaml"
    behavior_config: str = "config/behaviors/default.yaml"


class SystemConfig(BaseModel):
    """Top-level system configuration."""

    model_config = ConfigDict(frozen=True)

    operating_mode: OperatingMode = OperatingMode.CIVIL
    log_level: str = "INFO"
    safe_zones: tuple[SafeZone, ...] = ()
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @field_validator("operating_mode", mode="before")
    @classmethod
    def _coerce_operating_mode(cls, v: object) -> object:
        if isinstance(v, str):
            try:
                return OperatingMode[v]
            except KeyError:
                valid = ", ".join(m.name for m in OperatingMode)
                msg = f"Invalid operating mode '{v}'. Valid values: {valid}"
                raise ValueError(msg) from None
        return v


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load and parse a YAML file.

    Raises FileNotFoundError if the path does not exist.
    Raises yaml.YAMLError on malformed YAML.
    """
    with Path(path).open() as f:
        return yaml.safe_load(f) or {}


def load_system_config(path: str | Path) -> SystemConfig:
    """Load a SystemConfig from a YAML file with validation."""
    try:
        data = load_yaml(path)
        return SystemConfig(**data)
    except ValidationError as exc:
        msg = f"Invalid system config in {path}: {exc}"
        raise ValueError(msg) from exc


def load_hal_config(path: str | Path) -> HALConfig:
    """Load a HALConfig from a YAML file with validation."""
    try:
        data = load_yaml(path)
        return HALConfig(**data)
    except ValidationError as exc:
        msg = f"Invalid HAL config in {path}: {exc}"
        raise ValueError(msg) from exc
