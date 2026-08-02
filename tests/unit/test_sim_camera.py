"""Tests for the simulated camera provider.

Verifies:
1. SimCameraProvider satisfies the CameraHAL protocol
2. Frame generation produces correct shape, dtype, and content
3. Lifecycle guards: read before open, double open, close resets state
4. Unsupported formats are rejected
"""

from __future__ import annotations

import numpy as np
import pytest

from aerialmind.core.protocols import CameraHAL
from aerialmind.hal.camera.sim_provider import SimCameraProvider

# ---------------------------------------------------------------------------
# SimCameraProvider tests
# ---------------------------------------------------------------------------


class TestSimCameraProvider:
    @pytest.fixture()
    def provider(self) -> SimCameraProvider:
        return SimCameraProvider()

    @pytest.fixture()
    def config(self) -> dict[str, object]:
        return {"width": 640, "height": 480, "fps": 30, "format": "BGR"}

    def test_conforms_to_camera_hal_protocol(self, provider: SimCameraProvider) -> None:
        assert isinstance(provider, CameraHAL)

    def test_read_frame_before_open_returns_none(
        self, provider: SimCameraProvider,
    ) -> None:
        assert provider.read_frame() is None

    def test_open_succeeds(
        self, provider: SimCameraProvider, config: dict[str, object],
    ) -> None:
        provider.open(config)
        provider.close()

    def test_read_frame_returns_correct_shape(
        self, provider: SimCameraProvider, config: dict[str, object],
    ) -> None:
        provider.open(config)
        frame = provider.read_frame()
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        provider.close()

    def test_read_frame_returns_uint8(
        self, provider: SimCameraProvider, config: dict[str, object],
    ) -> None:
        provider.open(config)
        frame = provider.read_frame()
        assert frame is not None
        assert frame.dtype == np.uint8
        provider.close()

    def test_read_frame_is_deterministic(
        self, provider: SimCameraProvider, config: dict[str, object],
    ) -> None:
        provider.open(config)
        frame1 = provider.read_frame()
        frame2 = provider.read_frame()
        assert frame1 is not None
        assert frame2 is not None
        np.testing.assert_array_equal(frame1, frame2)
        provider.close()

    def test_seq_id_increments(
        self, provider: SimCameraProvider, config: dict[str, object],
    ) -> None:
        provider.open(config)
        provider.read_frame()
        assert provider._seq_id == 1
        provider.read_frame()
        assert provider._seq_id == 2
        provider.close()

    def test_get_intrinsics_has_expected_keys(
        self, provider: SimCameraProvider, config: dict[str, object],
    ) -> None:
        provider.open(config)
        intrinsics = provider.get_intrinsics()
        assert "fx" in intrinsics
        assert "fy" in intrinsics
        assert "cx" in intrinsics
        assert "cy" in intrinsics
        assert "dist_coeffs" in intrinsics
        provider.close()

    def test_close_resets_state(
        self, provider: SimCameraProvider, config: dict[str, object],
    ) -> None:
        provider.open(config)
        provider.close()
        assert provider.read_frame() is None

    def test_double_open_raises(
        self, provider: SimCameraProvider, config: dict[str, object],
    ) -> None:
        provider.open(config)
        with pytest.raises(RuntimeError, match="already opened"):
            provider.open(config)
        provider.close()

    def test_unsupported_format_raises(self, provider: SimCameraProvider) -> None:
        config = {"width": 640, "height": 480, "fps": 30, "format": "NV12"}
        with pytest.raises(ValueError, match="only supports BGR"):
            provider.open(config)
