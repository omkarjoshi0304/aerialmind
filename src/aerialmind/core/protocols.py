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

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aerialmind.core.types import (
    ActionDecision,
    BehaviorEvent,
    DecisionState,
    Detection,
    GPSFix,
    LinkStatus,
    NavState,
    OperatingMode,
    OpticalFlowReading,
    PoseResult,
    Recommendation,
    ThreatAssessment,
    TimestampedFrame,
    TimestampedIMU,
    Track,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


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

    def read_frame(self) -> NDArray[np.uint8] | None:
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

    def read(self) -> TimestampedIMU | None:
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

    def read(self) -> GPSFix | None:
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

    def read(self) -> OpticalFlowReading | None:
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

    def read_altitude(self) -> tuple[float, float] | None:
        """Read altitude. Returns (altitude_m, mono_ts) or None."""
        ...

    def close(self) -> None:
        """Release altimeter resources."""
        ...


# ---------------------------------------------------------------------------
# Vision Pipeline Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class ObjectDetectorInterface(Protocol):
    """Interface for object detection models (e.g., YOLOv10).

    Runs at 15 fps on every frame. The AcceleratorHAL handles the
    actual inference; this interface handles pre/post-processing.
    """

    def detect(self, frame: TimestampedFrame) -> list[Detection]:
        """Run detection on a single frame."""
        ...

    def get_class_names(self) -> list[str]:
        """Return the list of class names the current model can detect."""
        ...

    def swap_model(self, model_path: str) -> None:
        """Hot-swap the detection model (e.g., on mode switch).

        Called by the Mode Manager when switching between Military
        and Civil modes to load a different set of detection classes.
        """
        ...


@runtime_checkable
class PoseEstimatorInterface(Protocol):
    """Interface for skeletal pose estimation (e.g., YOLO-Pose).

    Runs at 5 fps (every 3rd frame). Only estimates poses on
    already-detected persons to save compute.
    """

    def estimate(
        self, frame: TimestampedFrame, detections: list[Detection]
    ) -> list[PoseResult]:
        """Estimate body poses for detected persons in the frame."""
        ...


@runtime_checkable
class TrackerInterface(Protocol):
    """Interface for multi-object tracking (e.g., ByteTrack).

    Assigns persistent track_ids across frames so the system can
    reason about individuals over time, not just per-frame blobs.
    """

    def update(
        self, detections: list[Detection], frame: TimestampedFrame
    ) -> list[Track]:
        """Update tracks with new detections from the current frame."""
        ...

    def reset(self) -> None:
        """Clear all track state.

        Called when camera view changes dramatically (e.g., drone
        pivots 180 degrees) to avoid false track continuity.
        """
        ...


@runtime_checkable
class BehaviorAnalyzerInterface(Protocol):
    """Interface for behavior pattern classification.

    Runs at 3 fps (every 5th frame). Watches tracks and poses
    over time to classify patterns: fights, weapon brandishing,
    crowd surges, loitering.
    """

    def analyze(
        self, tracks: list[Track], poses: list[PoseResult]
    ) -> list[BehaviorEvent]:
        """Classify behavior patterns from current tracks and poses."""
        ...


# ---------------------------------------------------------------------------
# Navigation Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class VIOEngineInterface(Protocol):
    """Interface for Visual-Inertial Odometry (e.g., ORB-SLAM3, Basalt).

    Runs in a separate C++ process (isolated Docker container) for
    crash safety. Communicates via shared memory for frame data.
    """

    def initialize(
        self, camera_intrinsics: dict[str, object], imu_noise: dict[str, object]
    ) -> None:
        """Initialize with camera calibration and IMU noise model.

        These values are hardware-specific — the HAL providers supply
        them via CameraHAL.get_intrinsics() and IMUHAL.get_noise_params().
        """
        ...

    def process_frame(
        self,
        frame: TimestampedFrame,
        imu_readings: list[TimestampedIMU],
    ) -> NavState | None:
        """Process a camera frame with IMU readings since the last frame.

        Returns None if tracking is lost (insufficient visual features).
        When this returns None, the EKF falls back to IMU-only prediction
        and optical flow if available.
        """
        ...

    def get_tracking_quality(self) -> float:
        """Return tracking quality, 0.0-1.0.

        Below ~0.3 = tracking is degraded (featureless terrain).
        Below ~0.1 = tracking is effectively lost.
        The CER controller monitors this to decide fallback strategies.
        """
        ...

    def reset(self) -> None:
        """Reset VIO state and map. Called on camera view changes."""
        ...


@runtime_checkable
class NavigationEKFInterface(Protocol):
    """Interface for the 15-state Extended Kalman Filter.

    The EKF fuses all sensor data into a single best estimate
    of the drone's state (position, velocity, attitude, biases).

    States: position(3) + velocity(3) + attitude(4 quaternion) +
            gyro_bias(3) + accel_bias(3) = 16 states
            (stored as 15 with quaternion constraint).
    """

    def predict(self, imu: TimestampedIMU) -> NavState:
        """IMU prediction step — runs at 200-400 Hz.

        Propagates state forward using accelerometer and gyroscope.
        This is the fast inner loop of the navigation system.
        """
        ...

    def update_gps(self, gps: GPSFix) -> NavState:
        """GPS measurement update — runs at 1-10 Hz.

        Only called when GPS integrity check passes (gps.valid == True).
        """
        ...

    def update_vio(self, vio_pose: NavState) -> NavState:
        """VIO measurement update — runs at 15 Hz.

        Primary position/attitude correction, especially in GPS-denied mode.
        """
        ...

    def update_baro(self, altitude: float, mono_ts: float) -> NavState:
        """Barometric altitude update — runs at 10 Hz.

        Constrains vertical drift. Less accurate than altimeter
        but available on all platforms.
        """
        ...

    def update_optical_flow(self, flow: OpticalFlowReading) -> NavState:
        """Optical flow velocity update.

        Fallback when VIO tracking is lost (featureless terrain).
        Provides X/Y velocity constraint with almost zero compute.
        """
        ...

    def update_altimeter(self, altitude: float, mono_ts: float) -> NavState:
        """Laser/radar altimeter update.

        Absolute ground-truth altitude — much more accurate than
        barometric altitude for vertical position estimation.
        """
        ...

    def get_state(self) -> NavState:
        """Return the current fused navigation state estimate."""
        ...


# ---------------------------------------------------------------------------
# Decision Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class DecisionEngineInterface(Protocol):
    """Interface for the rule-based decision engine.

    Deliberately NOT ML-based for the MVP — defense and law enforcement
    customers require every autonomous decision to be traceable to
    a specific ROE rule for legal and compliance reasons.
    """

    def process(
        self,
        behaviors: list[BehaviorEvent],
        tracks: list[Track],
        nav_state: NavState,
        link_status: LinkStatus,
    ) -> ActionDecision:
        """Process current intelligence and produce an action decision.

        This is the core loop: evidence in → auditable decision out.
        """
        ...

    def get_state(self) -> DecisionState:
        """Return the current decision state machine state."""
        ...

    def set_mode(self, mode: OperatingMode) -> None:
        """Switch operating mode (changes threat scoring weights)."""
        ...

    def load_roe(self, roe_path: str) -> None:
        """Load a signed ROE policy YAML file.

        ROE policies define the boundaries of autonomous action.
        The signature is verified before loading.
        """
        ...


@runtime_checkable
class TelemetryInterface(Protocol):
    """Interface for ground station communication.

    All data goes through AES-256-GCM encryption under the hood.
    """

    def send_scene_report(
        self,
        detections: list[Detection],
        tracks: list[Track],
        behaviors: list[BehaviorEvent],
        nav_state: NavState,
    ) -> bool:
        """Send current scene to the ground station. Returns success."""
        ...

    def send_alert(
        self,
        assessment: ThreatAssessment,
        recommendation: Recommendation,
    ) -> bool:
        """Send a threat alert to the operator. Returns success."""
        ...

    def receive_command(self) -> dict[str, object] | None:
        """Poll for operator commands. Returns None if no command."""
        ...
