"""Simulated camera provider for development and testing.

Generates synthetic BGR frames (horizontal gradient) without any
real camera hardware. Satisfies the CameraHAL protocol structurally.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SimCameraProvider:
    """Simulated camera that produces deterministic gradient frames."""

    def __init__(self) -> None:
        self._opened: bool = False
        self._width: int = 0
        self._height: int = 0
        self._fps: int = 0
        self._seq_id: int = 0
        self._frame: NDArray[np.uint8] | None = None

    def open(self, config: dict[str, object]) -> None:
        if self._opened:
            msg = "SimCameraProvider is already opened"
            raise RuntimeError(msg)

        fmt = config.get("format", "BGR")
        if fmt != "BGR":
            msg = f"SimCameraProvider only supports BGR format, got '{fmt}'"
            raise ValueError(msg)

        self._width = int(config.get("width", 1280))
        self._height = int(config.get("height", 720))
        self._fps = int(config.get("fps", 30))
        self._seq_id = 0

        row = np.linspace(0, 255, self._width, dtype=np.uint8)
        gray = np.tile(row, (self._height, 1))
        self._frame = np.stack([gray, gray, gray], axis=-1)

        self._opened = True

    def read_frame(self) -> NDArray[np.uint8] | None:
        if not self._opened or self._frame is None:
            return None
        self._seq_id += 1
        return self._frame

    def get_intrinsics(self) -> dict[str, object]:
        return {
            "fx": float(self._width),
            "fy": float(self._width),
            "cx": self._width / 2.0,
            "cy": self._height / 2.0,
            "dist_coeffs": (0.0, 0.0, 0.0, 0.0, 0.0),
        }

    def close(self) -> None:
        self._opened = False
        self._frame = None
        self._seq_id = 0
        self._width = 0
        self._height = 0
        self._fps = 0
