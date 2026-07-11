# Data Flow Diagrams

These sequence diagrams show exactly which module talks to which, in what order, and under what conditions. Read them top to bottom — each arrow is a message passed from one module to another.

## 3.1 Primary Frame Processing Pipeline

This is the main loop: a camera frame comes in, gets analyzed, and produces either an operator recommendation or an autonomous action.

```mermaid
sequenceDiagram
    participant CAM as Camera HAL
    participant ING as Camera Ingestor
    participant PRE as Frame Preprocessor
    participant DET as Object Detector
    participant POSE as Pose Estimator
    participant TRK as Multi-Object Tracker
    participant BEH as Behavior Analyzer
    participant DE as Decision Engine
    participant MAV as MAVLink Interface
    participant GCS as Ground Station

    loop Every frame (15 fps)
        CAM->>ING: Raw frame
        ING->>ING: Timestamp, calibrate
        ING->>PRE: TimestampedFrame
        PRE->>PRE: Resize, normalize, color convert
        PRE->>DET: InferenceTensor
        DET->>DET: YOLOv10 inference (AcceleratorHAL)
        DET->>TRK: List[Detection]
        TRK->>TRK: ByteTrack update
        TRK->>BEH: List[Track]
    end

    loop Every 3rd frame (5 fps)
        PRE->>POSE: InferenceTensor
        POSE->>POSE: YOLO-Pose inference
        POSE->>BEH: List[PoseResult]
    end

    loop Every 5th frame (3 fps)
        BEH->>BEH: Classify behaviors from tracks + poses
        BEH->>DE: List[BehaviorEvent]
    end

    loop Continuous
        DE->>DE: Assess threats, generate recommendations
        alt Operator Link Active
            DE->>MAV: Recommendation for operator approval
            MAV->>GCS: Encrypted telemetry
            GCS->>MAV: Operator decision
            MAV->>DE: Approved/rejected action
        else Operator Link Lost (CER Active)
            DE->>DE: Check ROE policy
            DE->>DE: Execute permitted autonomous action
        end
        DE->>MAV: Action command to flight controller
    end
```

**Key thing to notice**: three loops run at different speeds (15 fps, 5 fps, 3 fps) feeding into one continuous decision loop. This is the "deterministic scheduling" principle from the overview doc — expensive analysis runs less often without blocking the fast, cheap stuff.

---

## 3.2 Navigation Sensor Fusion Flow

This shows how GPS, IMU, camera, and barometer data combine into one position estimate — and what happens when GPS is jammed.

```mermaid
sequenceDiagram
    participant IMU as IMU HAL
    participant GPS as GPS HAL
    participant BARO as Barometer HAL
    participant CAM as Camera Ingestor
    participant VIO as VIO Engine
    participant EKF as Navigation EKF
    participant CER as CER Controller
    participant NAV as Path Planner
    participant DE as Decision Engine

    loop 200-400 Hz
        IMU->>EKF: TimestampedIMU (prediction step)
        EKF->>EKF: Propagate state estimate
    end

    loop 15 fps
        CAM->>VIO: TimestampedFrame
        IMU->>VIO: TimestampedIMU
        VIO->>VIO: ORB-SLAM3 / Basalt process
        VIO->>EKF: VIOPose (update step)
    end

    loop 1-10 Hz
        GPS->>EKF: TimestampedGPS (update step)
        GPS->>CER: GPS health data
        BARO->>EKF: BaroAlt (update step)
    end

    CER->>CER: Monitor GPS integrity + link health

    alt GPS Healthy & Link Active
        CER->>NAV: Normal navigation mode
        EKF->>NAV: Full NavState
        NAV->>DE: Trajectory following mission plan
    else GPS Denied or Spoofed
        CER->>CER: Switch to VIO-only mode
        CER->>NAV: GPS-denied constraints
        EKF->>NAV: VIO-fused NavState (higher uncertainty)
        NAV->>DE: Adjusted trajectory
    else GPS Denied AND Link Lost
        CER->>CER: Full CER mode
        CER->>NAV: Return to safe zone
        NAV->>DE: Safe-zone return trajectory
        DE->>DE: Autonomous ROE-bounded decisions only
    end
```

**Why IMU runs at 200-400 Hz while GPS runs at 1-10 Hz**: IMU (accelerometer + gyroscope) measures tiny, fast changes in motion — it needs to sample fast to keep the drone stable in flight. GPS gives an absolute position fix but updates slowly and can be jammed. The EKF uses IMU for the "predict" step every cycle and corrects drift with the slower, more reliable GPS/VIO "update" steps. This is the mathematical foundation of every modern autopilot.

---

## 3.3 Mode Switching Flow

This shows what happens when an operator switches from Military to Civil mode (or vice versa) mid-mission.

```mermaid
sequenceDiagram
    participant OP as Operator
    participant GCS as Ground Station
    participant MAV as MAVLink Interface
    participant MM as Mode Manager
    participant MR as Model Registry
    participant DET as Object Detector
    participant ROE as ROE Policy Store
    participant DE as Decision Engine

    OP->>GCS: Request mode switch (Military -> Civil)
    GCS->>MAV: ModeSwitch command + auth token
    MAV->>MM: ModeSwitch(CIVIL, token)
    MM->>MM: Validate auth token
    MM->>MM: Check preconditions (altitude, area, clearance)

    alt Preconditions met
        MM->>MR: Get civil mode model manifest
        MR->>MM: Model paths + classes
        MM->>DET: Load civil detection model
        DET->>DET: Swap TensorRT engine (warm up)
        MM->>ROE: Load civil ROE policy
        ROE->>DE: New ROE boundaries
        MM->>MAV: ModeStatus(CIVIL, success)
        MAV->>GCS: Mode switch confirmed
    else Preconditions failed
        MM->>MAV: ModeStatus(MILITARY, failure, reason)
        MAV->>GCS: Mode switch denied
    end
```

**Why an auth token is required**: Mode switching changes which ROE rules apply — civil mode should never accidentally have military-grade autonomous authority. Requiring authorization prevents accidental or unauthorized escalation.
