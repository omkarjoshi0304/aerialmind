# Module Specifications

Detailed breakdown of each module: what it does, its interfaces, internal components, and dependencies.

## 2.1 Hardware Abstraction Layer (HAL)

**Purpose**: Isolate all hardware-specific code behind stable interfaces so the rest of the system is hardware-agnostic.

**Responsibilities**:
- Provide uniform APIs for camera capture, IMU reads, GPS reads, barometer reads
- Abstract AI accelerator differences (TensorRT vs ONNX Runtime vs Hailo SDK)
- Abstract flight controller protocol differences

**Key Interfaces**:

| Interface | Input | Output |
|---|---|---|
| `CameraHAL` | resolution config, framerate | Raw frames (numpy ndarray) |
| `IMUHAL` | sample rate config | `IMUReading(accel, gyro, timestamp)` |
| `GPSHAL` | - | `GPSFix(lat, lon, alt, hdop, fix_type)` |
| `AcceleratorHAL` | ONNX model bytes, input tensor | Output tensor(s) |
| `FlightControllerHAL` | MAVLink command | MAVLink response |

**Internal Components**:
- `JetsonAccelerator` — wraps TensorRT engine build and inference
- `ONNXAccelerator` — wraps ONNX Runtime for portable fallback
- `HailoAccelerator` — wraps Hailo SDK for RPi5+Hailo deployments
- `V4L2Camera` / `CSICamera` / `GStreamerCamera` — camera backends
- `SerialIMU` / `SysfsIMU` — IMU backends
- Hardware discovery service that auto-detects available accelerators at startup

**Dependencies**: `pycuda`, `tensorrt`, `onnxruntime`, `hailo_platform` (each optional, loaded dynamically based on detected hardware)

**Why "optional, loaded dynamically"?** The same codebase runs on a $300 Jetson or a $50 Raspberry Pi. We don't want to require every dependency on every platform — the HAL checks what's actually installed and picks the right backend at startup.

---

## 2.2 Sensor Ingestion Layer

**Purpose**: Normalize, timestamp, and buffer raw sensor data from the HAL into a uniform internal format.

**Responsibilities**:
- Apply hardware-independent calibration (lens distortion, IMU bias)
- Synchronize multi-sensor timestamps to a common monotonic clock
- Provide ring buffers so downstream consumers never block sensor capture
- Detect sensor failures (frozen frames, IMU dropout) and emit health events

**Key Interfaces**:

| Interface | Input | Output |
|---|---|---|
| `CameraIngestor` | Raw frame from HAL | `TimestampedFrame(frame, mono_ts, seq_id)` |
| `IMUIngestor` | Raw IMU from HAL | `TimestampedIMU(accel, gyro, mono_ts)` |
| `GPSIngestor` | Raw GPS from HAL | `TimestampedGPS(fix, mono_ts, valid)` |

**Internal Components**:
- `TimeSync` — correlates hardware timestamps to system monotonic clock
- `SensorHealthMonitor` — watchdog per sensor, emits `SensorDegraded` / `SensorLost` events
- `CalibrationStore` — loads per-unit calibration files (camera intrinsics, IMU noise params)

**Why timestamp synchronization matters**: Camera, IMU, and GPS all report on their own clocks with different latencies. If we naively use "time received" the EKF gets slightly wrong data and drifts. `TimeSync` maps every reading to one monotonic clock so sensor fusion math is correct.

**Dependencies**: `numpy`, `scipy`

---

## 2.3 Vision Pipeline

**Purpose**: Transform raw camera frames into structured scene understanding — detected objects, human poses, tracked identities, and behavioral classifications.

**Responsibilities**:
- Preprocess frames (resize, normalize, color convert) for inference
- Run object detection at target framerate
- Run pose estimation on detected persons
- Maintain multi-object tracks across frames
- Classify behaviors from track histories and pose sequences
- Provide mode-specific detection model selection (military vs civil)

**Key Interfaces**:

| Interface | Input | Output |
|---|---|---|
| `FramePreprocessor` | `TimestampedFrame` | `InferenceTensor` |
| `ObjectDetector` | `InferenceTensor` | `List[Detection(bbox, class_id, confidence)]` |
| `PoseEstimator` | `InferenceTensor`, `List[Detection]` | `List[PoseResult(keypoints, detection)]` |
| `MultiObjectTracker` | `List[Detection]`, frame | `List[Track(track_id, bbox, velocity, age)]` |
| `BehaviorAnalyzer` | `List[Track]`, `List[PoseResult]` | `List[BehaviorEvent(type, confidence, tracks)]` |

**Internal Components**:
- `ModelRegistry` — stores model metadata (path, input shape, classes, mode compatibility)
- `DetectorFactory` — instantiates the correct detector based on current mode and available accelerator
- `ByteTrackAdapter` — wraps ByteTrack for multi-object tracking
- `BehaviorClassifier` — rule-based + ML classifier for behavior patterns (fight detection, crowd surge, weapon brandishing)
- `AerialPerspectiveCorrector` — compensates for oblique drone camera angles in bounding box and pose coordinates

**Dependencies**: `ultralytics` (YOLOv10, YOLO-Pose), `opencv-python`, `numpy`, `supervision`

**Framerate Budget** (Jetson Orin Nano, 640x480 input):

| Stage | Target Latency | Frequency |
|---|---|---|
| Preprocessing | <2ms | Every frame |
| Object Detection | <30ms | Every frame (15 fps) |
| Pose Estimation | <20ms | Every 3rd frame (5 fps) |
| Tracking | <5ms | Every frame |
| Behavior Analysis | <10ms | Every 5th frame (3 fps) |

**Why not run everything at full framerate?** Pose estimation and behavior analysis are more expensive than raw detection, and behaviors (a fight, a weapon raise) unfold over seconds — not milliseconds. Running them less often saves compute for the things that need to be instant (detection, tracking) without losing accuracy on the things that don't.

---

## 2.4 Navigation Engine

**Purpose**: Maintain a continuous, accurate estimate of the drone's position and orientation, even under GPS denial, and plan safe trajectories.

**Responsibilities**:
- Run Visual-Inertial Odometry for GPS-denied localization
- Fuse all available navigation sources (GPS, VIO, barometer, IMU) via EKF
- Detect GPS spoofing/jamming and trigger CER mode
- Plan return-to-safe-zone paths when disconnected
- Provide navigation state to the Decision Engine

**Key Interfaces**:

| Interface | Input | Output |
|---|---|---|
| `VIOEngine` | `TimestampedFrame`, `TimestampedIMU` | `VIOPose(position, orientation, velocity, covariance)` |
| `NavigationEKF` | `VIOPose`, `TimestampedGPS`, `TimestampedIMU`, `BaroAlt` | `NavState(pos, vel, att, pos_uncertainty)` |
| `PathPlanner` | `NavState`, waypoints, obstacles | `Trajectory(waypoints, velocities, timestamps)` |
| `CERController` | `GPSHealth`, `LinkHealth`, `NavState` | `CERState(mode, action, safe_zone_bearing)` |

**Internal Components**:
- `ORBSLAMWrapper` — C++ shared library with Python bindings via pybind11
- `BasaltWrapper` — alternative VIO backend, same interface
- `EKFCore` — C++ implementation of 15-state EKF (position, velocity, attitude, gyro bias, accel bias)
- `GPSIntegrityChecker` — statistical tests for GPS spoofing
- `SafeZoneManager` — stores pre-programmed safe return coordinates

**Dependencies**: `orb_slam3` (C++ with pybind11 bindings), `numpy`, `scipy`, `pyproj`

Full detail on this module: [CER Design](05-cer-design.md)

---

## 2.5 Communication Layer

**Purpose**: Handle all external communication — flight controller commands, ground station telemetry, and operator data links — with encryption and link health monitoring.

**Responsibilities**:
- Send/receive MAVLink messages to/from the flight controller
- Stream encrypted telemetry (detections, nav state, video thumbnails) to ground station
- Monitor link quality and detect disconnection
- Buffer critical messages during link loss for later burst transmission
- Enforce message authentication (prevent command injection)

**Key Interfaces**:

| Interface | Input | Output |
|---|---|---|
| `MAVLinkInterface` | `MAVCommand` | `MAVResponse` / heartbeat stream |
| `TelemetryManager` | `SceneReport`, `NavState` | Serialized + encrypted telemetry packets |
| `CryptoModule` | plaintext bytes | ciphertext bytes (AES-256-GCM) |
| `LinkHealthMonitor` | heartbeat timing | `LinkStatus(connected, latency, rssi, quality)` |

**Internal Components**:
- `MAVLinkRouter` — multiplexes MAVLink streams (FC on serial, GCS on UDP)
- `MessageQueue` — priority queue for outbound messages, persists to flash during link loss
- `SessionKeyManager` — handles key exchange and rotation
- `HeartbeatWatchdog` — fires `LinkLost` event after configurable timeout (default: 3 seconds)

**Dependencies**: `pymavlink` or `mavsdk`, `cryptography`, `protobuf`

Full detail on security: [Security Architecture](04-security.md)

---

## 2.6 Decision Engine

**Purpose**: Synthesize all sensor and analysis data into actionable decisions, provide recommendations to the human operator, and execute autonomous actions within ROE boundaries when disconnected.

**Responsibilities**:
- Continuously assess threat level from vision pipeline outputs
- Generate prioritized recommendations
- When operator link is active: present recommendations, await approval
- When operator link is lost: execute pre-approved ROE actions autonomously
- Log all decisions with full audit trail for post-mission review
- Respect mode-specific rules (military ROE vs civil operating procedures)

**Key Interfaces**:

| Interface | Input | Output |
|---|---|---|
| `ThreatAssessor` | `List[BehaviorEvent]`, `List[Track]`, `NavState` | `ThreatAssessment(level, threats, confidence)` |
| `RecommendationGenerator` | `ThreatAssessment`, `MissionContext` | `List[Recommendation(action, priority, rationale)]` |
| `AutonomousActionController` | `Recommendation`, `ROEPolicy`, `LinkStatus` | `ActionDecision(approved, action, authority)` |
| `AuditLogger` | any decision event | persistent log entry |

**Internal Components**:
- `ROEPolicyEngine` — evaluates whether an autonomous action is permitted by current ROE
- `DecisionStateMachine` — tracks system state (IDLE, MONITORING, TRACKING, ALERTING, RETURNING)
- `MissionContextStore` — holds current mission parameters
- `ActionExecutor` — translates approved actions into MAVLink commands or telemetry alerts

**Dependencies**: No external ML dependencies; rule-based engine with configurable policy files (YAML/JSON)

**Why rule-based, not ML?** Defense and government customers require explainability — every autonomous decision must trace back to a specific, auditable rule. See [Decision Engine Design](06-decision-engine.md) for the full reasoning.

---

## 2.7 Mode Manager & Orchestrator

**Purpose**: Manage runtime mode switching (Military/Civil) and coordinate the startup, shutdown, and lifecycle of all modules.

**Responsibilities**:
- Switch detection models and ROE policies when mode changes
- Validate mode-switch preconditions (e.g., cannot switch to military mode without proper authorization token)
- Orchestrate module initialization order and dependency injection
- Provide system health dashboard data
- Handle graceful degradation when modules fail

**Key Interfaces**:

| Interface | Input | Output |
|---|---|---|
| `ModeManager` | `ModeSwitch(target_mode, auth_token)` | `ModeStatus(current, available_models)` |
| `Orchestrator` | system config | running system with all modules wired |

**Internal Components**:
- `ModuleRegistry` — tracks all registered modules and their health
- `ConfigLoader` — reads mission config, ROE files, model manifests
- `SystemHealthAggregator` — collects health from all modules, computes system-level health score

**Dependencies**: `pydantic` (config validation), `pyyaml`
