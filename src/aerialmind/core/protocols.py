"""AerialMind module protocol interfaces.

Every module is built against these interfaces, not against each other's
implementation. This is the mechanism that makes "plug in any drone hardware"
work in Python without inheritance.

Uses typing.Protocol (structural typing):
    - Any class with the right methods satisfies the Protocol automatically.
    - No inheritance or import required from the implementing side.
    - @runtime_checkable allows isinstance() checks for dynamic loading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from numpy.typing import NDArray

    import numpy as np

from aerialmind.core.types import OpticalFlowReading, TimestampedIMU


# ---------------------------------------------------------------------------
# HAL Protocols — Hardware Abstraction Layer
# ---------------------------------------------------------------------------


@runtime_checkable
class CameraHAL(Protocol):
    """Interface for any camera source.

    Implementations: CSICameraProvider, V4L2CameraProvider,
    GStreamerCameraProvider, SimCameraProvider.
    """

    def open(self, config: dict[str, object]) -> None:
        """Initialize the camera with platform-specific config."""
        ...

    def read_frame(self) -> Optional[NDArray[np.uint8]]:
        """Capture a single frame. Returns None on failure."""
        ...

    def get_intrinsics(self) -> dict[str, object]:
        """Return camera calibration matrix and distortion coefficients.

        The VIO engine needs these to convert 2D pixel coordinates
        into 3D positions.
        """
        ...

    def close(self) -> None:
        """Release camera resources."""
        ...


@runtime_checkable
class AcceleratorHAL(Protocol):
    """Interface for any AI inference hardware.

    Implementations: TensorRTProvider (Jetson), HailoProvider,
    ONNXRuntimeProvider, CPUFallbackProvider.

    Startup priority: TensorRT > Hailo > ONNX(CUDA) > ONNX(CPU).
    """

    def load_model(self, model_path: str, input_shapes: dict[str, object]) -> str:
        """Load an ONNX model and compile to native format. Returns model_id."""
        ...

    def infer(
        self, model_id: str, inputs: dict[str, NDArray[np.float32]]
    ) -> dict[str, NDArray[np.float32]]:
        """Run synchronous inference. Returns output tensors by name."""
        ...

    def get_capabilities(self) -> dict[str, object]:
        """Report hardware capabilities.

        Returns: {"precision": ["fp32", "fp16", "int8"],
                  "max_batch": N, "device_name": str}
        """
        ...

    def unload_model(self, model_id: str) -> None:
        """Release a loaded model's resources."""
        ...


@runtime_checkable
class IMUHAL(Protocol):
    """Interface for any inertial measurement unit.

    Implementations: SerialIMUProvider, SPIIMUProvider, SimIMUProvider.
    """

    def open(self, config: dict[str, object]) -> None:
        """Initialize the IMU with platform-specific config."""
        ...

    def read(self) -> Optional[TimestampedIMU]:
        """Read a single IMU sample. Returns None on failure."""
        ...

    def get_noise_params(self) -> dict[str, object]:
        """Return IMU noise characteristics for EKF tuning.

        The EKF uses these to weight IMU readings — a cheap MEMS
        IMU gets less trust than a tactical-grade fiber-optic gyro.

        Returns: {"accel_noise": float, "gyro_noise": float,
                  "accel_bias_instability": float,
                  "gyro_bias_instability": float}
        """
        ...

    def close(self) -> None:
        """Release IMU resources."""
        ...


@runtime_checkable
class GPSHAL(Protocol):
    """Interface for any GPS receiver.

    Implementations: SerialGPSProvider, SimGPSProvider.
    """

    def open(self, config: dict[str, object]) -> None:
        """Initialize the GPS receiver."""
        ...

    def read(self) -> Optional["GPSFix"]:
        """Read a GPS fix. Returns None if no fix available."""
        ...

    def close(self) -> None:
        """Release GPS resources."""
        ...


@runtime_checkable
class OpticalFlowHAL(Protocol):
    """Interface for a downward-facing optical flow sensor.

    Provides raw X/Y velocity even when ORB-SLAM3 fails over
    featureless terrain (desert, water, night). Almost zero compute.

    Implementations: PX4FlowProvider, SimOpticalFlowProvider.
    """

    def open(self, config: dict[str, object]) -> None:
        """Initialize the optical flow sensor."""
        ...

    def read(self) -> Optional[OpticalFlowReading]:
        """Read an optical flow measurement. Returns None on failure."""
        ...

    def close(self) -> None:
        """Release sensor resources."""
        ...


@runtime_checkable
class AltimeterHAL(Protocol):
    """Interface for a laser/radar altimeter.

    Provides absolute ground-truth altitude. Barometers drift with
    weather; a LiDAR altimeter doesn't, taking strain off the EKF's
    vertical uncertainty estimate.

    Implementations: LidarAltimeterProvider, SimAltimeterProvider.
    """

    def open(self, config: dict[str, object]) -> None:
        """Initialize the altimeter."""
        ...

    def read_altitude(self) -> Optional[tuple[float, float]]:
        """Read altitude. Returns (altitude_m, mono_ts) or None."""
        ...

    def close(self) -> None:
        """Release altimeter resources."""
        ...
