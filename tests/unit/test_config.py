"""Tests for configuration models and YAML loader.

Verifies:
1. Pydantic models construct and validate correctly
2. Frozen config objects reject attribute assignment
3. YAML loader handles valid files, missing files, and malformed YAML
4. Real config template files parse without errors
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from aerialmind.core.config import (
    AcceleratorConfig,
    CameraConfig,
    GPSConfig,
    HALConfig,
    IMUConfig,
    PathsConfig,
    SafeZone,
    SystemConfig,
    load_hal_config,
    load_system_config,
    load_yaml,
)
from aerialmind.core.types import OperatingMode

# ---------------------------------------------------------------------------
# HAL config model tests
# ---------------------------------------------------------------------------


class TestCameraConfig:
    def test_valid_construction(self) -> None:
        cam = CameraConfig(provider="csi")
        assert cam.provider == "csi"

    def test_defaults(self) -> None:
        cam = CameraConfig(provider="sim")
        assert cam.width == 1280
        assert cam.height == 720
        assert cam.fps == 30
        assert cam.format == "NV12"

    def test_frozen(self) -> None:
        cam = CameraConfig(provider="csi")
        with pytest.raises(ValidationError):
            cam.provider = "v4l2"  # type: ignore[misc]


class TestAcceleratorConfig:
    def test_valid_construction(self) -> None:
        acc = AcceleratorConfig(provider="tensorrt")
        assert acc.provider == "tensorrt"
        assert acc.precision == "fp16"

    def test_frozen(self) -> None:
        acc = AcceleratorConfig(provider="cpu")
        with pytest.raises(ValidationError):
            acc.precision = "fp32"  # type: ignore[misc]


class TestIMUConfig:
    def test_valid_construction(self) -> None:
        imu = IMUConfig(provider="serial")
        assert imu.sample_rate_hz == 400

    def test_frozen(self) -> None:
        imu = IMUConfig(provider="serial")
        with pytest.raises(ValidationError):
            imu.baudrate = 0  # type: ignore[misc]


class TestGPSConfig:
    def test_valid_construction(self) -> None:
        gps = GPSConfig(provider="serial")
        assert gps.baudrate == 115200

    def test_frozen(self) -> None:
        gps = GPSConfig(provider="serial")
        with pytest.raises(ValidationError):
            gps.device = "/dev/null"  # type: ignore[misc]


class TestHALConfig:
    def test_full_config(self) -> None:
        hal = HALConfig(
            platform="jetson_orin_nano",
            camera=CameraConfig(provider="csi"),
            accelerator=AcceleratorConfig(provider="tensorrt"),
            imu=IMUConfig(provider="serial"),
            gps=GPSConfig(provider="serial"),
        )
        assert hal.platform == "jetson_orin_nano"
        assert hal.imu is not None
        assert hal.gps is not None

    def test_optional_imu_and_gps(self) -> None:
        hal = HALConfig(
            platform="sim",
            camera=CameraConfig(provider="sim"),
            accelerator=AcceleratorConfig(provider="cpu"),
        )
        assert hal.imu is None
        assert hal.gps is None

    def test_frozen(self) -> None:
        hal = HALConfig(
            platform="sim",
            camera=CameraConfig(provider="sim"),
            accelerator=AcceleratorConfig(provider="cpu"),
        )
        with pytest.raises(ValidationError):
            hal.platform = "other"  # type: ignore[misc]

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            HALConfig(platform="test")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# System config model tests
# ---------------------------------------------------------------------------


class TestSafeZone:
    def test_valid_construction(self) -> None:
        sz = SafeZone(lat=37.7749, lon=-122.4194, radius_m=500.0, label="Alpha")
        assert sz.lat == pytest.approx(37.7749)

    def test_frozen(self) -> None:
        sz = SafeZone(lat=0.0, lon=0.0, radius_m=100.0, label="Test")
        with pytest.raises(ValidationError):
            sz.label = "changed"  # type: ignore[misc]


class TestPathsConfig:
    def test_defaults(self) -> None:
        paths = PathsConfig()
        assert paths.hal_config == "config/hal/sim.yaml"
        assert paths.roe_policy == "config/roe/civil_standard.yaml"

    def test_custom_paths(self) -> None:
        paths = PathsConfig(hal_config="/custom/hal.yaml")
        assert paths.hal_config == "/custom/hal.yaml"

    def test_frozen(self) -> None:
        paths = PathsConfig()
        with pytest.raises(ValidationError):
            paths.hal_config = "other"  # type: ignore[misc]


class TestSystemConfig:
    def test_defaults(self) -> None:
        cfg = SystemConfig()
        assert cfg.operating_mode == OperatingMode.CIVIL
        assert cfg.log_level == "INFO"
        assert cfg.safe_zones == ()

    def test_full_config(self) -> None:
        cfg = SystemConfig(
            operating_mode=OperatingMode.MILITARY,
            log_level="DEBUG",
            safe_zones=[
                SafeZone(lat=37.0, lon=-122.0, radius_m=500.0, label="Base"),
            ],
        )
        assert cfg.operating_mode == OperatingMode.MILITARY
        assert len(cfg.safe_zones) == 1

    def test_operating_mode_enum_coercion(self) -> None:
        cfg = SystemConfig(operating_mode="CIVIL")  # type: ignore[arg-type]
        assert cfg.operating_mode == OperatingMode.CIVIL

    def test_safe_zones_are_tuple(self) -> None:
        cfg = SystemConfig(
            safe_zones=[
                SafeZone(lat=0.0, lon=0.0, radius_m=100.0, label="A"),
                SafeZone(lat=1.0, lon=1.0, radius_m=200.0, label="B"),
            ],
        )
        assert isinstance(cfg.safe_zones, tuple)
        assert len(cfg.safe_zones) == 2

    def test_frozen(self) -> None:
        cfg = SystemConfig()
        with pytest.raises(ValidationError):
            cfg.log_level = "DEBUG"  # type: ignore[misc]

    def test_invalid_operating_mode_raises(self) -> None:
        with pytest.raises(ValidationError):
            SystemConfig(operating_mode="INVALID")  # type: ignore[arg-type]

    def test_operating_mode_case_sensitive(self) -> None:
        with pytest.raises(ValidationError):
            SystemConfig(operating_mode="civil")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# YAML loader tests
# ---------------------------------------------------------------------------


class TestLoadYaml:
    def test_load_valid_yaml(self, tmp_path: Path) -> None:
        f = tmp_path / "test.yaml"
        f.write_text("key: value\nnested:\n  a: 1\n")
        data = load_yaml(f)
        assert data == {"key": "value", "nested": {"a": 1}}

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_yaml("/nonexistent/path.yaml")

    def test_malformed_yaml_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text(":\n  - :\n    bad: [unterminated")
        with pytest.raises(yaml.YAMLError):
            load_yaml(f)

    def test_empty_yaml_returns_empty_dict(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.yaml"
        f.write_text("")
        data = load_yaml(f)
        assert data == {}


class TestLoadSystemConfig:
    def test_load_system_yaml(self, tmp_path: Path) -> None:
        f = tmp_path / "system.yaml"
        f.write_text(
            "operating_mode: MILITARY\n"
            "log_level: DEBUG\n"
            "safe_zones:\n"
            "  - lat: 37.0\n"
            "    lon: -122.0\n"
            "    radius_m: 500\n"
            '    label: "Base"\n',
        )
        cfg = load_system_config(f)
        assert cfg.operating_mode == OperatingMode.MILITARY
        assert cfg.log_level == "DEBUG"
        assert len(cfg.safe_zones) == 1

    def test_invalid_config_raises_value_error(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("operating_mode: INVALID_MODE\n")
        with pytest.raises(ValueError, match="Invalid system config"):
            load_system_config(f)

    def test_missing_fields_use_defaults(self, tmp_path: Path) -> None:
        f = tmp_path / "minimal.yaml"
        f.write_text("{}\n")
        cfg = load_system_config(f)
        assert cfg.operating_mode == OperatingMode.CIVIL
        assert cfg.log_level == "INFO"


class TestLoadHALConfig:
    def test_load_full_hal_config(self, tmp_path: Path) -> None:
        f = tmp_path / "hal.yaml"
        f.write_text(
            "platform: jetson\n"
            "camera:\n"
            "  provider: csi\n"
            "accelerator:\n"
            "  provider: tensorrt\n"
            "imu:\n"
            "  provider: serial\n"
            "gps:\n"
            "  provider: serial\n",
        )
        hal = load_hal_config(f)
        assert hal.platform == "jetson"
        assert hal.imu is not None
        assert hal.gps is not None

    def test_load_sim_config_no_imu_gps(self, tmp_path: Path) -> None:
        f = tmp_path / "sim.yaml"
        f.write_text(
            "platform: sim\n"
            "camera:\n"
            "  provider: sim\n"
            "accelerator:\n"
            "  provider: cpu\n",
        )
        hal = load_hal_config(f)
        assert hal.imu is None
        assert hal.gps is None

    def test_missing_platform_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text(
            "camera:\n"
            "  provider: sim\n"
            "accelerator:\n"
            "  provider: cpu\n",
        )
        with pytest.raises(ValueError, match="Invalid HAL config"):
            load_hal_config(f)

    def test_missing_camera_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("platform: test\n")
        with pytest.raises(ValueError, match="Invalid HAL config"):
            load_hal_config(f)


# ---------------------------------------------------------------------------
# Smoke tests against real config template files
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestLoadRealConfigFiles:
    def test_system_yaml(self) -> None:
        cfg = load_system_config(REPO_ROOT / "config" / "system.yaml")
        assert cfg.operating_mode == OperatingMode.CIVIL
        assert len(cfg.safe_zones) == 2

    def test_jetson_hal(self) -> None:
        hal = load_hal_config(REPO_ROOT / "config" / "hal" / "jetson_orin_nano.yaml")
        assert hal.platform == "jetson_orin_nano"
        assert hal.camera.provider == "csi"
        assert hal.imu is not None
        assert hal.gps is not None

    def test_rpi5_hal(self) -> None:
        hal = load_hal_config(REPO_ROOT / "config" / "hal" / "rpi5_hailo.yaml")
        assert hal.platform == "rpi5_hailo"
        assert hal.accelerator.precision == "int8"
        assert hal.imu is not None

    def test_sim_hal(self) -> None:
        hal = load_hal_config(REPO_ROOT / "config" / "hal" / "sim.yaml")
        assert hal.platform == "sim"
        assert hal.accelerator.provider == "cpu"
        assert hal.imu is None
        assert hal.gps is None
