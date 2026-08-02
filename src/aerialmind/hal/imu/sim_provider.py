"""Simulated IMU provider for development and testing.

Returns stable-hover readings (gravity on Z, zero rotation) without
any real IMU hardware. Satisfies the IMUHAL protocol structurally.
"""

from __future__ import annotations

import time

from aerialmind.core.types import TimestampedIMU

_GRAVITY_Z = -9.81


class SimIMUProvider:
    """Simulated IMU that returns perfect stable-hover readings."""

    def __init__(self) -> None:
        self._opened: bool = False
        self._sample_rate_hz: int = 0

    def open(self, config: dict[str, object]) -> None:
        if self._opened:
            msg = "SimIMUProvider is already opened"
            raise RuntimeError(msg)
        self._sample_rate_hz = int(config.get("sample_rate_hz", 400))
        self._opened = True

    def read(self) -> TimestampedIMU | None:
        if not self._opened:
            return None
        return TimestampedIMU(
            accel=(0.0, 0.0, _GRAVITY_Z),
            gyro=(0.0, 0.0, 0.0),
            mono_ts=time.monotonic(),
        )

    def get_noise_params(self) -> dict[str, object]:
        return {
            "accel_noise": 0.01,
            "gyro_noise": 0.001,
            "accel_bias_instability": 0.0001,
            "gyro_bias_instability": 0.00001,
        }

    def close(self) -> None:
        self._opened = False
        self._sample_rate_hz = 0
