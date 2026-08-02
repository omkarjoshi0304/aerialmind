"""Tests for the simulated GPS provider.

Verifies:
1. SimGPSProvider satisfies the GPSHAL protocol
2. Returns a valid 3D fix at the expected position
3. Lifecycle guards: read before open, double open, close resets
"""

from __future__ import annotations

import pytest

from aerialmind.core.protocols import GPSHAL
from aerialmind.core.types import GPSFix
from aerialmind.hal.gps.sim_provider import SimGPSProvider

# ---------------------------------------------------------------------------
# SimGPSProvider tests
# ---------------------------------------------------------------------------


class TestSimGPSProvider:
    @pytest.fixture()
    def provider(self) -> SimGPSProvider:
        return SimGPSProvider()

    @pytest.fixture()
    def config(self) -> dict[str, object]:
        return {"device": "/dev/null", "baudrate": 115200}

    def test_conforms_to_gps_hal_protocol(self, provider: SimGPSProvider) -> None:
        assert isinstance(provider, GPSHAL)

    def test_read_before_open_returns_none(self, provider: SimGPSProvider) -> None:
        assert provider.read() is None

    def test_read_returns_gps_fix(
        self, provider: SimGPSProvider, config: dict[str, object],
    ) -> None:
        provider.open(config)
        fix = provider.read()
        assert isinstance(fix, GPSFix)
        provider.close()

    def test_fix_is_valid(
        self, provider: SimGPSProvider, config: dict[str, object],
    ) -> None:
        provider.open(config)
        fix = provider.read()
        assert fix is not None
        assert fix.valid is True
        provider.close()

    def test_fix_type_is_3d(
        self, provider: SimGPSProvider, config: dict[str, object],
    ) -> None:
        provider.open(config)
        fix = provider.read()
        assert fix is not None
        assert fix.fix_type == 3
        provider.close()

    def test_latitude_and_longitude_are_plausible(
        self, provider: SimGPSProvider, config: dict[str, object],
    ) -> None:
        provider.open(config)
        fix = provider.read()
        assert fix is not None
        assert -90.0 <= fix.latitude <= 90.0
        assert -180.0 <= fix.longitude <= 180.0
        provider.close()

    def test_hdop_is_excellent(
        self, provider: SimGPSProvider, config: dict[str, object],
    ) -> None:
        provider.open(config)
        fix = provider.read()
        assert fix is not None
        assert fix.hdop <= 1.0
        provider.close()

    def test_mono_ts_is_positive(
        self, provider: SimGPSProvider, config: dict[str, object],
    ) -> None:
        provider.open(config)
        fix = provider.read()
        assert fix is not None
        assert fix.mono_ts > 0
        provider.close()

    def test_mono_ts_increases(
        self, provider: SimGPSProvider, config: dict[str, object],
    ) -> None:
        provider.open(config)
        f1 = provider.read()
        f2 = provider.read()
        assert f1 is not None
        assert f2 is not None
        assert f2.mono_ts >= f1.mono_ts
        provider.close()

    def test_close_resets_state(
        self, provider: SimGPSProvider, config: dict[str, object],
    ) -> None:
        provider.open(config)
        provider.close()
        assert provider.read() is None

    def test_double_open_raises(
        self, provider: SimGPSProvider, config: dict[str, object],
    ) -> None:
        provider.open(config)
        with pytest.raises(RuntimeError, match="already opened"):
            provider.open(config)
        provider.close()
