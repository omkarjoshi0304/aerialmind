"""Integration tests for HAL providers.

Verifies:
1. Discovery service runs without error
2. Factory builds working providers from sim config YAML
3. Full lifecycle: open -> read -> close works end-to-end
4. Protocol conformance on real provider classes (not inline stubs)
5. All sim providers return valid data types
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aerialmind.core.config import (
    AcceleratorConfig,
    CameraConfig,
    GPSConfig,
    HALConfig,
    IMUConfig,
)
from aerialmind.core.protocols import GPSHAL, IMUHAL, AcceleratorHAL, CameraHAL
from aerialmind.core.types import GPSFix, TimestampedIMU
from aerialmind.hal.accelerator.cpu_provider import CPUFallbackProvider
from aerialmind.hal.camera.sim_provider import SimCameraProvider
from aerialmind.hal.discovery import HardwareCapabilities, discover_hardware
from aerialmind.hal.factory import HALProviders, load_hal_from_yaml
from aerialmind.hal.gps.sim_provider import SimGPSProvider
from aerialmind.hal.imu.sim_provider import SimIMUProvider

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Discovery integration
# ---------------------------------------------------------------------------


class TestDiscoveryIntegration:
    def test_discover_hardware_runs_without_error(self) -> None:
        caps = discover_hardware()
        assert isinstance(caps, HardwareCapabilities)

    def test_discovery_result_is_consistent(self) -> None:
        caps1 = discover_hardware()
        caps2 = discover_hardware()
        assert caps1.has_onnxruntime == caps2.has_onnxruntime
        assert caps1.platform_hint == caps2.platform_hint


# ---------------------------------------------------------------------------
# Sim provider lifecycle (from real YAML)
# ---------------------------------------------------------------------------


class TestSimProviderLifecycle:
    @pytest.fixture()
    def hal_providers(self) -> HALProviders:
        providers = load_hal_from_yaml(REPO_ROOT / "config" / "hal" / "sim.yaml")
        yield providers
        providers.camera.close()

    def test_camera_produces_frames(self, hal_providers: HALProviders) -> None:
        frame = hal_providers.camera.read_frame()
        assert frame is not None
        assert frame.shape == (720, 1280, 3)

    def test_camera_frame_is_uint8(self, hal_providers: HALProviders) -> None:
        frame = hal_providers.camera.read_frame()
        assert frame is not None
        assert frame.dtype == np.uint8

    def test_camera_intrinsics_has_focal_length(
        self, hal_providers: HALProviders,
    ) -> None:
        intrinsics = hal_providers.camera.get_intrinsics()
        assert "fx" in intrinsics
        assert "fy" in intrinsics

    def test_accelerator_reports_cpu(self, hal_providers: HALProviders) -> None:
        caps = hal_providers.accelerator.get_capabilities()
        assert caps["device_name"] == "cpu"

    def test_accelerator_precision_is_fp32(self, hal_providers: HALProviders) -> None:
        caps = hal_providers.accelerator.get_capabilities()
        assert "fp32" in caps["precision"]

    def test_imu_is_none_for_sim_config(self, hal_providers: HALProviders) -> None:
        assert hal_providers.imu is None

    def test_gps_is_none_for_sim_config(self, hal_providers: HALProviders) -> None:
        assert hal_providers.gps is None


# ---------------------------------------------------------------------------
# Full provider set (sim camera + cpu accel + sim imu + sim gps)
# ---------------------------------------------------------------------------


class TestSimProviderWithIMUAndGPS:
    @pytest.fixture()
    def hal_providers(self) -> HALProviders:
        from aerialmind.hal.factory import create_hal_providers

        config = HALConfig(
            platform="sim",
            camera=CameraConfig(provider="sim", format="BGR"),
            accelerator=AcceleratorConfig(provider="cpu"),
            imu=IMUConfig(provider="sim"),
            gps=GPSConfig(provider="sim"),
        )
        providers = create_hal_providers(config)
        yield providers
        providers.camera.close()
        if providers.imu:
            providers.imu.close()
        if providers.gps:
            providers.gps.close()

    def test_imu_read_returns_timestamped_imu(
        self, hal_providers: HALProviders,
    ) -> None:
        assert hal_providers.imu is not None
        reading = hal_providers.imu.read()
        assert isinstance(reading, TimestampedIMU)

    def test_imu_accel_has_gravity(self, hal_providers: HALProviders) -> None:
        assert hal_providers.imu is not None
        reading = hal_providers.imu.read()
        assert reading is not None
        assert reading.accel[2] == pytest.approx(-9.81)

    def test_gps_read_returns_gps_fix(self, hal_providers: HALProviders) -> None:
        assert hal_providers.gps is not None
        fix = hal_providers.gps.read()
        assert isinstance(fix, GPSFix)

    def test_gps_fix_is_valid(self, hal_providers: HALProviders) -> None:
        assert hal_providers.gps is not None
        fix = hal_providers.gps.read()
        assert fix is not None
        assert fix.valid is True

    def test_full_lifecycle(self, hal_providers: HALProviders) -> None:
        frame = hal_providers.camera.read_frame()
        assert frame is not None

        assert hal_providers.imu is not None
        imu_reading = hal_providers.imu.read()
        assert imu_reading is not None

        assert hal_providers.gps is not None
        gps_fix = hal_providers.gps.read()
        assert gps_fix is not None


# ---------------------------------------------------------------------------
# Protocol conformance on real provider classes
# ---------------------------------------------------------------------------


class TestProtocolConformanceOnRealProviders:
    def test_sim_camera_is_camera_hal(self) -> None:
        assert isinstance(SimCameraProvider(), CameraHAL)

    def test_cpu_fallback_is_accelerator_hal(self) -> None:
        assert isinstance(CPUFallbackProvider(), AcceleratorHAL)

    def test_sim_imu_is_imu_hal(self) -> None:
        assert isinstance(SimIMUProvider(), IMUHAL)

    def test_sim_gps_is_gps_hal(self) -> None:
        assert isinstance(SimGPSProvider(), GPSHAL)


# ---------------------------------------------------------------------------
# Factory from YAML
# ---------------------------------------------------------------------------


class TestFactoryFromYaml:
    def test_sim_yaml_produces_working_providers(self) -> None:
        providers = load_hal_from_yaml(REPO_ROOT / "config" / "hal" / "sim.yaml")
        frame = providers.camera.read_frame()
        assert frame is not None
        assert frame.shape[2] == 3
        providers.camera.close()

    def test_factory_opens_camera_automatically(self) -> None:
        providers = load_hal_from_yaml(REPO_ROOT / "config" / "hal" / "sim.yaml")
        frame = providers.camera.read_frame()
        assert frame is not None
        providers.camera.close()
