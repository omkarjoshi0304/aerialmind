# API Contracts

This document defines the exact Python types and interfaces that modules use to talk to each other. These will become real code in `src/aerialmind/core/types.py` and `src/aerialmind/core/protocols.py` during Feature 2.

## Why Define This Before Writing Code?

These contracts are the "wiring diagram" of the whole system. Every module — Vision Pipeline, Navigation Engine, Decision Engine — is built against these interfaces, not against each other's implementation details. This is what lets us build modules independently (in separate feature branches) and have them work together correctly on the first try.

## Core Enums

```python
class OperatingMode(Enum):
    MILITARY = auto()
    CIVIL = auto()

class ThreatLevel(Enum):
    NONE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4

class CERState(Enum):
    NOMINAL = auto()
    GPS_DEGRADED = auto()
    GPS_DENIED = auto()
    LINK_DEGRADED = auto()
    LINK_LOST = auto()
    CER_PARTIAL = auto()
    CER_FULL = auto()
    SAFE_ZONE_RETURN = auto()
    EMERGENCY_LAND = auto()
    LANDED = auto()

class DecisionState(Enum):
    IDLE = auto()
    MONITORING = auto()
    TRACKING = auto()
    ALERTING = auto()
    AUTONOMOUS_ACTION = auto()
    RETURNING = auto()
    LANDED = auto()

class ResourcePriority(Enum):
    """QoS priority levels for compute resource allocation.
    
    In NOMINAL mode, vision gets most compute. When CER triggers,
    navigation becomes life-or-death — the Orchestrator's QoS Manager
    uses these to dynamically throttle vision and give VIO/EKF
    100% of the compute needed to keep the drone in the air.
    """
    CRITICAL = 0       # Navigation during CER — never throttle
    HIGH = 1           # Vision in NOMINAL — full framerate
    NORMAL = 2         # Pose estimation — can drop to lower fps
    LOW = 3            # Behavior analysis — can pause entirely
    BACKGROUND = 4     # Logging, telemetry — best-effort
```

## Value Objects (Data Types)

These use Python's `@dataclass(frozen=True)` — meaning once created, they cannot be modified. This is intentional: sensor readings and computed states are historical facts at the moment they're created. Immutability prevents an entire category of bugs where one module accidentally mutates data another module is still using.

Types containing numpy arrays (`TimestampedFrame`, `PoseResult`) use `eq=False` because numpy arrays don't support element-wise `==` in a boolean context. Identity comparison (`is`) is the correct semantic for these — two frames are distinct captures even if pixel data matches.

All container fields use immutable types: `tuple` instead of `list`, and typed frozen dataclasses (`ThreatSignal`, `ActionConstraints`) instead of `dict`. This guarantees deep immutability, not just shallow field reassignment prevention.

```python
@dataclass(frozen=True, eq=False)
class TimestampedFrame:
    frame: NDArray[np.uint8]        # H x W x C
    mono_ts: float                  # monotonic timestamp (seconds)
    seq_id: int
    width: int
    height: int

@dataclass(frozen=True)
class TimestampedIMU:
    accel: tuple[float, float, float]   # m/s^2, body frame
    gyro: tuple[float, float, float]    # rad/s, body frame
    mono_ts: float

@dataclass(frozen=True)
class GPSFix:
    latitude: float
    longitude: float
    altitude_msl: float
    hdop: float
    fix_type: int                   # 0=no fix, 2=2D, 3=3D, 4=DGPS, 5=RTK
    num_satellites: int
    mono_ts: float
    valid: bool

@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2 normalized
    class_id: int
    class_name: str
    confidence: float

@dataclass(frozen=True, eq=False)
class PoseResult:
    keypoints: NDArray[np.float32]   # Kx3 (x, y, confidence)
    detection: Detection

@dataclass(frozen=True)
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    velocity: tuple[float, float]     # px/frame
    age: int                          # frames since first seen
    class_id: int
    class_name: str
    confidence: float

@dataclass(frozen=True)
class BehaviorEvent:
    event_type: str                  # "fight", "weapon_brandish", "crowd_surge", etc.
    confidence: float
    involved_tracks: tuple[int, ...]  # track_ids (tuple for immutability)
    location_frame: tuple[float, float]  # center x, y in frame coords
    mono_ts: float

@dataclass(frozen=True)
class OpticalFlowReading:
    """Raw X/Y velocity from a downward-facing optical flow sensor (e.g., PX4Flow).
    
    Works even when ORB-SLAM3 can't find visual features (desert, water, night).
    Almost zero compute cost — a cheap hardware fallback for GPS-denied flight.
    """
    vx: float                       # m/s, body-frame X
    vy: float                       # m/s, body-frame Y
    quality: int                    # 0-255, sensor-reported confidence
    ground_distance: float          # meters, from integrated rangefinder
    mono_ts: float

@dataclass(frozen=True)
class NavState:
    position: tuple[float, float, float]         # lat, lon, alt (or local NED if GPS-denied)
    velocity: tuple[float, float, float]         # m/s NED
    attitude_wxyz: tuple[float, float, float, float]  # quaternion (w, x, y, z)
    position_uncertainty: tuple[float, float, float]  # 1-sigma meters (N, E, D)
    coordinate_frame: str                        # "WGS84" or "LOCAL_NED"
    mono_ts: float

@dataclass(frozen=True)
class LinkStatus:
    connected: bool
    latency_ms: float
    rssi_dbm: float | None
    quality_pct: float              # 0-100
    last_heartbeat_ts: float

@dataclass(frozen=True)
class ThreatSignal:
    track_id: int
    type: str                       # "weapon", "aggressive_behavior", etc.
    confidence: float

@dataclass(frozen=True)
class ActionConstraints:
    max_duration_sec: float | None = None
    max_deviation_m: float | None = None
    min_altitude_m: float | None = None

@dataclass(frozen=True)
class ThreatAssessment:
    level: ThreatLevel
    score: float                    # 0-100
    threats: tuple[ThreatSignal, ...]
    mono_ts: float

@dataclass(frozen=True)
class Recommendation:
    action: str                     # "TRACK_CLOSELY", "ALERT_AUTHORITIES", "RETURN_TO_BASE"
    priority: int                   # 1 (highest) - 5 (lowest)
    rationale: str
    constraints: ActionConstraints
    roe_rule_id: str | None

@dataclass(frozen=True)
class ActionDecision:
    approved: bool
    action: str
    authority: str                  # "OPERATOR", "ROE_AUTONOMOUS", "CER_OVERRIDE"
    constraints: ActionConstraints
    audit_id: str
```

## Module Protocols (Abstract Interfaces)

Python's `Protocol` (from `typing`) defines "structural typing" — any class that implements these methods automatically satisfies the interface, without needing to explicitly inherit from it. This is what allows the `TensorRTProvider`, `HailoProvider`, and `CPUFallbackProvider` to all be swapped in as an `AcceleratorHAL` without any of them needing to know about each other.

```python
class CameraHAL(Protocol):
    def open(self, config: dict) -> None: ...
    def read_frame(self) -> Optional[NDArray[np.uint8]]: ...
    def get_intrinsics(self) -> dict: ...
    def close(self) -> None: ...

class AcceleratorHAL(Protocol):
    def load_model(self, model_path: str, input_shapes: dict) -> str: ...
    """Returns model_id"""

    def infer(self, model_id: str, inputs: dict[str, NDArray]) -> dict[str, NDArray]: ...
    """Synchronous inference. Returns output tensors by name."""

    def get_capabilities(self) -> dict: ...
    """Returns {precision: [fp32, fp16, int8], max_batch: N, device_name: str}"""

    def unload_model(self, model_id: str) -> None: ...

class IMUHAL(Protocol):
    def open(self, config: dict) -> None: ...
    def read(self) -> Optional[TimestampedIMU]: ...
    def get_noise_params(self) -> dict: ...
    def close(self) -> None: ...

class GPSHAL(Protocol):
    def open(self, config: dict) -> None: ...
    def read(self) -> Optional[GPSFix]: ...
    def close(self) -> None: ...

class OpticalFlowHAL(Protocol):
    """Downward-facing optical flow sensor (e.g., PX4Flow).
    
    Provides raw X/Y velocity even when ORB-SLAM3 fails (featureless terrain).
    Almost zero compute — a cheap hardware safety net for GPS-denied flight.
    """
    def open(self, config: dict) -> None: ...
    def read(self) -> Optional[OpticalFlowReading]: ...
    def close(self) -> None: ...

class AltimeterHAL(Protocol):
    """Laser/radar altimeter for absolute Z-axis altitude.
    
    Barometers drift with weather. A cheap LiDAR altimeter gives ground-truth
    height, taking massive strain off the EKF's vertical uncertainty estimate.
    """
    def open(self, config: dict) -> None: ...
    def read_altitude(self) -> Optional[tuple[float, float]]: ...
    """Returns (altitude_m, mono_ts) or None if no reading."""
    def close(self) -> None: ...

class ObjectDetectorInterface(Protocol):
    def detect(self, frame: TimestampedFrame) -> list[Detection]: ...
    def get_class_names(self) -> list[str]: ...
    def swap_model(self, model_path: str) -> None: ...

class PoseEstimatorInterface(Protocol):
    def estimate(self, frame: TimestampedFrame,
                 detections: list[Detection]) -> list[PoseResult]: ...

class TrackerInterface(Protocol):
    def update(self, detections: list[Detection],
               frame: TimestampedFrame) -> list[Track]: ...
    def reset(self) -> None: ...

class BehaviorAnalyzerInterface(Protocol):
    def analyze(self, tracks: list[Track],
                poses: list[PoseResult]) -> list[BehaviorEvent]: ...

class VIOEngineInterface(Protocol):
    def initialize(self, camera_intrinsics: dict,
                   imu_noise: dict) -> None: ...
    def process_frame(self, frame: TimestampedFrame,
                      imu_readings: list[TimestampedIMU]) -> Optional[NavState]: ...
    def get_tracking_quality(self) -> float: ...
    def reset(self) -> None: ...

class NavigationEKFInterface(Protocol):
    def predict(self, imu: TimestampedIMU) -> NavState: ...
    def update_gps(self, gps: GPSFix) -> NavState: ...
    def update_vio(self, vio_pose: NavState) -> NavState: ...
    def update_baro(self, altitude: float, mono_ts: float) -> NavState: ...
    def update_optical_flow(self, flow: OpticalFlowReading) -> NavState: ...
    def update_altimeter(self, altitude: float, mono_ts: float) -> NavState: ...
    def get_state(self) -> NavState: ...

class DecisionEngineInterface(Protocol):
    def process(self, behaviors: list[BehaviorEvent],
                tracks: list[Track],
                nav_state: NavState,
                link_status: LinkStatus) -> ActionDecision: ...
    def get_state(self) -> DecisionState: ...
    def set_mode(self, mode: OperatingMode) -> None: ...
    def load_roe(self, roe_path: str) -> None: ...

class TelemetryInterface(Protocol):
    def send_scene_report(self, detections: list[Detection],
                          tracks: list[Track],
                          behaviors: list[BehaviorEvent],
                          nav_state: NavState) -> bool: ...
    def send_alert(self, assessment: ThreatAssessment,
                   recommendation: Recommendation) -> bool: ...
    def receive_command(self) -> Optional[dict]: ...
```

## How These Map to Feature 2

When we implement Feature 2 (Core Types & Protocols), these exact definitions become:
- `src/aerialmind/core/types.py` — all the enums and dataclasses above
- `src/aerialmind/core/protocols.py` — all the Protocol interfaces above

Every module built afterward (Vision Pipeline in Feature 5, Navigation Engine in Feature 6, etc.) imports from these two files and implements the relevant protocols. This is why Feature 2 comes first in the implementation order — everything else depends on it.
