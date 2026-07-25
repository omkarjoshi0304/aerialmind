# AerialMind Project Roadmap

Feature breakdown with user stories. Each feature maps to a branch, each user story maps to a commit.

## Branching Strategy

```
main (protected — PR review required)
  │
  ├── doc/architecture     → Feature 1
  ├── feature/core-types   → Feature 2
  ├── feature/message-bus-config → Feature 3
  ├── feature/hal-layer    → Feature 4
  ├── feature/vision-pipeline → Feature 5
  ├── feature/navigation-engine → Feature 6
  ├── feature/comms-layer  → Feature 7
  ├── feature/decision-engine → Feature 8
  ├── feature/orchestrator → Feature 9
  ├── feature/docker-deployment → Feature 10
  └── feature/security     → Feature 11
```

---

## Phase 0: Repository & Documentation

### Feature 0: Repository Setup ✅

> Branch: `main` (direct push, one-time bootstrap)

| # | User Story | Commit | Status |
|---|---|---|---|
| 0.1 | Initialize repo with folder structure, .gitignore, LICENSE, pyproject.toml | `Initialize AerialMind project structure` | ✅ |
| 0.2 | Add README with project overview and badges | `Add project README` | ✅ |

### Feature 1: Architecture Documentation

> Branch: `doc/architecture` → PR → merge to main

| # | User Story | Commit | Status |
|---|---|---|---|
| 1.1 | Add system architecture overview with Mermaid diagrams | `Add system architecture doc` | ✅ |
| 1.2 | Add module specifications (HAL, Vision, Nav, Comms, Decision) | `Add module specification docs` | ✅ |
| 1.3 | Add data flow diagrams (frame pipeline, nav fusion, mode switching) | `Add data flow diagrams` | ✅ |
| 1.4 | Add security architecture doc | `Add security architecture doc` | ✅ |
| 1.5 | Add CER (soft-kill resistance) design doc | `Add CER design doc` | ✅ |
| 1.6 | Add decision engine design doc | `Add decision engine design doc` | ✅ |
| 1.7 | Add deployment architecture doc | `Add deployment architecture doc` | ✅ |
| 1.8 | Add API contracts doc (types + protocols) | `Add API contracts doc` | ✅ |
| 1.9 | Add project roadmap and contributing guide | `Add roadmap and contributing guide` | ✅ |

---

## Phase 1: Foundation

### Feature 2: Core Types & Protocols ✅

> Branch: `feature/core-types`

| # | User Story | Commit | Status |
|---|---|---|---|
| 2.1 | Add core enums (OperatingMode, ThreatLevel, CERState, DecisionState, ResourcePriority) | `Add core enum types` | ✅ |
| 2.2 | Add sensor data types (TimestampedFrame, TimestampedIMU, GPSFix, OpticalFlowReading) | `Add sensor data types` | ✅ |
| 2.3 | Add vision data types (Detection, PoseResult, Track, BehaviorEvent) | `Add vision data types` | ✅ |
| 2.4 | Add navigation data types (NavState, LinkStatus, ThreatAssessment) | `Add navigation and decision data types` | ✅ |
| 2.5 | Add HAL protocol interfaces (CameraHAL, AcceleratorHAL, IMUHAL, GPSHAL, OpticalFlowHAL, AltimeterHAL) | `Add HAL protocol interfaces` | ✅ |
| 2.6 | Add vision protocol interfaces (ObjectDetector, Tracker, BehaviorAnalyzer) | `Add vision protocol interfaces` | ✅ |
| 2.7 | Add navigation + decision protocol interfaces | `Add navigation and decision protocols` | ✅ |
| 2.8 | Add unit tests for all types (38 tests) | `Add tests for core types` | ✅ |

### Feature 3: Message Bus & Config

> Branch: `feature/message-bus-config`

| # | User Story | Commit | Status |
|---|---|---|---|
| 3.1 | Add internal pub-sub message bus | `Add internal event message bus` | |
| 3.2 | Add Pydantic config models and YAML loader | `Add configuration loader with validation` | |
| 3.3 | Add system.yaml and HAL config templates | `Add config templates` | |
| 3.4 | Add tests for bus and config | `Add tests for message bus and config` | |

---

## Phase 2: Hardware Abstraction

### Feature 4: HAL Implementation

> Branch: `feature/hal-layer`

| # | User Story | Commit | Status |
|---|---|---|---|
| 4.1 | Add hardware discovery service (probe for CUDA, Hailo, cameras) | `Add hardware discovery service` | |
| 4.2 | Add SimCameraProvider (generates test frames) | `Add simulated camera provider` | |
| 4.3 | Add CPUFallbackProvider (ONNX Runtime on CPU) | `Add CPU fallback accelerator` | |
| 4.4 | Add SimIMUProvider and SimGPSProvider | `Add simulated IMU and GPS providers` | |
| 4.5 | Add HAL config YAML loader per platform | `Add HAL config loader` | |
| 4.6 | Add integration test: all sim providers working together | `Add HAL integration tests` | |

---

## Phase 3: Vision Pipeline

### Feature 5: Vision Pipeline

> Branch: `feature/vision-pipeline`

| # | User Story | Commit | Status |
|---|---|---|---|
| 5.1 | Add frame preprocessor (resize, normalize, color convert) | `Add frame preprocessor` | |
| 5.2 | Add model registry (scan /models for manifests) | `Add model registry` | |
| 5.3 | Add YOLOv10 object detector via AcceleratorHAL | `Add YOLOv10 object detector` | |
| 5.4 | Add ByteTrack multi-object tracker | `Add ByteTrack tracker adapter` | |
| 5.5 | Add YOLO-Pose estimator | `Add pose estimator` | |
| 5.6 | Add behavior analyzer (fight, weapon brandish rules) | `Add behavior analyzer` | |
| 5.7 | Add vision pipeline orchestrator (chain all stages) | `Add vision pipeline orchestrator` | |
| 5.8 | Add end-to-end test with recorded VisDrone video | `Add vision pipeline E2E test` | |

---

## Phase 4: Navigation

### Feature 6: Navigation Engine

> Branch: `feature/navigation-engine`

| # | User Story | Commit | Status |
|---|---|---|---|
| 6.1 | Add EKF core (15-state Extended Kalman Filter) | `Add EKF sensor fusion core` | |
| 6.2 | Add VIO engine wrapper (ORB-SLAM3 interface) | `Add VIO engine wrapper` | |
| 6.3 | Add GPS integrity checker (5-layer detection) | `Add GPS integrity checker` | |
| 6.4 | Add safe zone manager | `Add safe zone manager` | |
| 6.5 | Add path planner | `Add path planner` | |
| 6.6 | Add CER controller state machine | `Add CER controller` | |
| 6.7 | Add test: simulated GPS loss → VIO handoff | `Add navigation integration tests` | |

---

## Phase 5: Communication

### Feature 7: Communication Layer

> Branch: `feature/comms-layer`

| # | User Story | Commit | Status |
|---|---|---|---|
| 7.1 | Add MAVLink interface (pymavlink wrapper) | `Add MAVLink interface` | |
| 7.2 | Add link health monitor (heartbeat watchdog) | `Add link health monitor` | |
| 7.3 | Add telemetry manager (scene reports, alerts) | `Add telemetry manager` | |
| 7.4 | Add crypto module (AES-256-GCM encryption) | `Add crypto module` | |
| 7.5 | Add test with PX4 SITL | `Add comms integration test` | |

---

## Phase 6: Decision Engine

### Feature 8: Decision Engine

> Branch: `feature/decision-engine`

| # | User Story | Commit | Status |
|---|---|---|---|
| 8.1 | Add threat assessor (weighted composite scoring) | `Add threat assessor` | |
| 8.2 | Add ROE policy parser (YAML + signature verification) | `Add ROE policy engine` | |
| 8.3 | Add recommendation generator | `Add recommendation generator` | |
| 8.4 | Add autonomous action controller | `Add autonomous action controller` | |
| 8.5 | Add decision state machine | `Add decision state machine` | |
| 8.6 | Add audit logger (encrypted, tamper-evident) | `Add audit logger` | |
| 8.7 | Add test with synthetic threat scenarios | `Add decision engine tests` | |

---

## Phase 7: Integration

### Feature 9: Mode Manager & Orchestrator

> Branch: `feature/orchestrator`

| # | User Story | Commit | Status |
|---|---|---|---|
| 9.1 | Add mode manager (military/civil switching with auth) | `Add mode manager` | |
| 9.2 | Add system orchestrator (module lifecycle, DI) | `Add system orchestrator` | |
| 9.3 | Add system health aggregator | `Add health aggregator` | |
| 9.4 | Add end-to-end integration test | `Add E2E integration test` | |

---

## Phase 8: Deployment & Security

### Feature 10: Docker Deployment

> Branch: `feature/docker-deployment`

| # | User Story | Commit | Status |
|---|---|---|---|
| 10.1 | Add Dockerfile.core and Dockerfile.hal | `Add core Docker images` | |
| 10.2 | Add Dockerfile.vio and Dockerfile.telemetry | `Add VIO and telemetry Docker images` | |
| 10.3 | Add docker-compose.yaml with shared memory + IPC | `Add docker-compose stack` | |
| 10.4 | Add auto-detect installer script | `Add installer script` | |

### Feature 11: Security Hardening

> Branch: `feature/security`

| # | User Story | Commit | Status |
|---|---|---|---|
| 11.1 | Add model signing and verification (Ed25519) | `Add model signing` | |
| 11.2 | Add MAVLink message authentication (HMAC-SHA256) | `Add MAVLink auth` | |
| 11.3 | Add encrypted audit logs | `Add encrypted logging` | |
| 11.4 | Add anti-tamper response | `Add anti-tamper system` | |

---

**Total: 12 features, ~60 user stories**
