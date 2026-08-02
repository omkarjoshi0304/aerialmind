"""Simulated GPS provider for development and testing.

Returns a fixed position (safe zone Alpha from system.yaml) without
any real GPS hardware. Satisfies the GPSHAL protocol structurally.
"""

from __future__ import annotations

import time

from aerialmind.core.types import GPSFix


class SimGPSProvider:
    """Simulated GPS that returns a fixed position."""

    def __init__(self) -> None:
        self._opened: bool = False

    def open(self, config: dict[str, object]) -> None:
        if self._opened:
            msg = "SimGPSProvider is already opened"
            raise RuntimeError(msg)
        self._opened = True

    def read(self) -> GPSFix | None:
        if not self._opened:
            return None
        return GPSFix(
            latitude=37.7749,
            longitude=-122.4194,
            altitude_msl=100.0,
            hdop=1.0,
            fix_type=3,
            num_satellites=12,
            mono_ts=time.monotonic(),
            valid=True,
        )

    def close(self) -> None:
        self._opened = False
