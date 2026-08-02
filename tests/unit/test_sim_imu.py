"""Tests for the simulated IMU provider.

Verifies:
1. SimIMUProvider satisfies the IMUHAL protocol
2. Returns stable-hover readings (gravity on Z, zero rotation)
3. Lifecycle guards: read before open, double open, close resets
4. Noise parameters are present and positive
"""

from __future__ import annotations

import pytest

from aerialmind.core.protocols import IMUHAL
from aerialmind.core.types import TimestampedIMU
from aerialmind.hal.imu.sim_provider import SimIMUProvider

# ---------------------------------------------------------------------------
# SimIMUProvider tests
# ---------------------------------------------------------------------------


class TestSimIMUProvider:
    @pytest.fixture()
    def provider(self) -> SimIMUProvider:
        return SimIMUProvider()

    @pytest.fixture()
    def config(self) -> dict[str, object]:
        return {"sample_rate_hz": 400, "device": "/dev/null", "baudrate": 921600}

    def test_conforms_to_imu_hal_protocol(self, provider: SimIMUProvider) -> None:
        assert isinstance(provider, IMUHAL)

    def test_read_before_open_returns_none(self, provider: SimIMUProvider) -> None:
        assert provider.read() is None

    def test_read_returns_timestamped_imu(
        self, provider: SimIMUProvider, config: dict[str, object],
    ) -> None:
        provider.open(config)
        reading = provider.read()
        assert isinstance(reading, TimestampedIMU)
        provider.close()

    def test_accel_is_gravity_vector(
        self, provider: SimIMUProvider, config: dict[str, object],
    ) -> None:
        provider.open(config)
        reading = provider.read()
        assert reading is not None
        assert reading.accel == (0.0, 0.0, -9.81)
        provider.close()

    def test_gyro_is_zero(
        self, provider: SimIMUProvider, config: dict[str, object],
    ) -> None:
        provider.open(config)
        reading = provider.read()
        assert reading is not None
        assert reading.gyro == (0.0, 0.0, 0.0)
        provider.close()

    def test_mono_ts_is_positive(
        self, provider: SimIMUProvider, config: dict[str, object],
    ) -> None:
        provider.open(config)
        reading = provider.read()
        assert reading is not None
        assert reading.mono_ts > 0
        provider.close()

    def test_mono_ts_increases(
        self, provider: SimIMUProvider, config: dict[str, object],
    ) -> None:
        provider.open(config)
        r1 = provider.read()
        r2 = provider.read()
        assert r1 is not None
        assert r2 is not None
        assert r2.mono_ts >= r1.mono_ts
        provider.close()

    def test_get_noise_params_has_expected_keys(
        self, provider: SimIMUProvider,
    ) -> None:
        params = provider.get_noise_params()
        assert "accel_noise" in params
        assert "gyro_noise" in params
        assert "accel_bias_instability" in params
        assert "gyro_bias_instability" in params

    def test_get_noise_params_values_are_positive(
        self, provider: SimIMUProvider,
    ) -> None:
        params = provider.get_noise_params()
        for key, value in params.items():
            assert isinstance(value, float), f"{key} should be float"
            assert value > 0, f"{key} should be positive"

    def test_close_resets_state(
        self, provider: SimIMUProvider, config: dict[str, object],
    ) -> None:
        provider.open(config)
        provider.close()
        assert provider.read() is None

    def test_double_open_raises(
        self, provider: SimIMUProvider, config: dict[str, object],
    ) -> None:
        provider.open(config)
        with pytest.raises(RuntimeError, match="already opened"):
            provider.open(config)
        provider.close()
