"""Tests for the HAL provider factory.

Verifies:
1. HALProviders dataclass is frozen
2. Factory creates correct provider types from config
3. Unknown provider strings raise ValueError with available list
4. Factory loads from real YAML files
5. Unimplemented providers are rejected
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aerialmind.core.config import (
    AcceleratorConfig,
    CameraConfig,
    GPSConfig,
    HALConfig,
    IMUConfig,
)
from aerialmind.core.protocols import GPSHAL, IMUHAL, AcceleratorHAL, CameraHAL
from aerialmind.hal.factory import HALProviders, create_hal_providers, load_hal_from_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# HALProviders dataclass
# ---------------------------------------------------------------------------


class TestHALProviders:
    def test_is_frozen(self) -> None:
        config = HALConfig(
            platform="sim",
            camera=CameraConfig(provider="sim", format="BGR"),
            accelerator=AcceleratorConfig(provider="cpu"),
        )
        providers = create_hal_providers(config)
        with pytest.raises(AttributeError):
            providers.camera = None  # type: ignore[misc]
        providers.camera.close()


# ---------------------------------------------------------------------------
# create_hal_providers
# ---------------------------------------------------------------------------


class TestCreateHALProviders:
    @pytest.fixture()
    def sim_config(self) -> HALConfig:
        return HALConfig(
            platform="sim",
            camera=CameraConfig(provider="sim", format="BGR"),
            accelerator=AcceleratorConfig(provider="cpu"),
        )

    @pytest.fixture()
    def full_config(self) -> HALConfig:
        return HALConfig(
            platform="sim",
            camera=CameraConfig(provider="sim", format="BGR"),
            accelerator=AcceleratorConfig(provider="cpu"),
            imu=IMUConfig(provider="sim"),
            gps=GPSConfig(provider="sim"),
        )

    def test_returns_hal_providers(self, sim_config: HALConfig) -> None:
        providers = create_hal_providers(sim_config)
        assert isinstance(providers, HALProviders)
        providers.camera.close()

    def test_camera_is_camera_hal(self, sim_config: HALConfig) -> None:
        providers = create_hal_providers(sim_config)
        assert isinstance(providers.camera, CameraHAL)
        providers.camera.close()

    def test_accelerator_is_accelerator_hal(self, sim_config: HALConfig) -> None:
        providers = create_hal_providers(sim_config)
        assert isinstance(providers.accelerator, AcceleratorHAL)
        providers.camera.close()

    def test_imu_is_none_when_not_configured(self, sim_config: HALConfig) -> None:
        providers = create_hal_providers(sim_config)
        assert providers.imu is None
        providers.camera.close()

    def test_gps_is_none_when_not_configured(self, sim_config: HALConfig) -> None:
        providers = create_hal_providers(sim_config)
        assert providers.gps is None
        providers.camera.close()

    def test_imu_is_imu_hal_when_configured(self, full_config: HALConfig) -> None:
        providers = create_hal_providers(full_config)
        assert isinstance(providers.imu, IMUHAL)
        providers.camera.close()
        if providers.imu:
            providers.imu.close()
        if providers.gps:
            providers.gps.close()

    def test_gps_is_gps_hal_when_configured(self, full_config: HALConfig) -> None:
        providers = create_hal_providers(full_config)
        assert isinstance(providers.gps, GPSHAL)
        providers.camera.close()
        if providers.imu:
            providers.imu.close()
        if providers.gps:
            providers.gps.close()

    def test_unknown_camera_provider_raises(self) -> None:
        config = HALConfig(
            platform="test",
            camera=CameraConfig(provider="nonexistent", format="BGR"),
            accelerator=AcceleratorConfig(provider="cpu"),
        )
        with pytest.raises(ValueError, match="Unknown camera provider"):
            create_hal_providers(config)

    def test_unknown_accelerator_provider_raises(self) -> None:
        config = HALConfig(
            platform="test",
            camera=CameraConfig(provider="sim", format="BGR"),
            accelerator=AcceleratorConfig(provider="nonexistent"),
        )
        with pytest.raises(ValueError, match="Unknown accelerator provider"):
            create_hal_providers(config)

    def test_unknown_imu_provider_raises(self) -> None:
        config = HALConfig(
            platform="test",
            camera=CameraConfig(provider="sim", format="BGR"),
            accelerator=AcceleratorConfig(provider="cpu"),
            imu=IMUConfig(provider="nonexistent"),
        )
        with pytest.raises(ValueError, match="Unknown IMU provider"):
            create_hal_providers(config)

    def test_unknown_gps_provider_raises(self) -> None:
        config = HALConfig(
            platform="test",
            camera=CameraConfig(provider="sim", format="BGR"),
            accelerator=AcceleratorConfig(provider="cpu"),
            gps=GPSConfig(provider="nonexistent"),
        )
        with pytest.raises(ValueError, match="Unknown GPS provider"):
            create_hal_providers(config)

    def test_error_message_lists_available_providers(self) -> None:
        config = HALConfig(
            platform="test",
            camera=CameraConfig(provider="bad", format="BGR"),
            accelerator=AcceleratorConfig(provider="cpu"),
        )
        with pytest.raises(ValueError, match="Available: sim"):
            create_hal_providers(config)


# ---------------------------------------------------------------------------
# load_hal_from_yaml
# ---------------------------------------------------------------------------


class TestLoadHALFromYaml:
    def test_loads_sim_yaml(self) -> None:
        providers = load_hal_from_yaml(REPO_ROOT / "config" / "hal" / "sim.yaml")
        assert isinstance(providers, HALProviders)
        assert isinstance(providers.camera, CameraHAL)
        assert isinstance(providers.accelerator, AcceleratorHAL)
        assert providers.imu is None
        assert providers.gps is None
        providers.camera.close()

    def test_jetson_yaml_raises_for_unimplemented_providers(self) -> None:
        with pytest.raises(ValueError, match="Unknown camera provider 'csi'"):
            load_hal_from_yaml(
                REPO_ROOT / "config" / "hal" / "jetson_orin_nano.yaml",
            )
