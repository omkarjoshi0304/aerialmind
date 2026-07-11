# AerialMind

**AI-powered drone surveillance software with Cognitive Edge Resilience**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Pre--Alpha-orange.svg)]()

AerialMind is a lightweight, hardware-agnostic AI software platform that turns any standard drone into an intelligent surveillance system. It runs on edge compute modules (NVIDIA Jetson, Raspberry Pi + Hailo, or CPU fallback) and communicates with any MAVLink-compatible flight controller.

## Key Features

- **Dual-Mode Operation** — Military (weapon detection, deployment mapping) and Civil (anomaly detection, traffic monitoring) modes, switchable at runtime
- **Cognitive Edge Resilience (CER)** — Survives GPS jamming and RF soft-kill attacks by switching to Visual-Inertial Odometry for GPS-denied navigation
- **Smart Disconnect** — When operator link is lost, the AI follows pre-approved Rules of Engagement (ROE) to act autonomously within safe boundaries
- **Hardware-Agnostic** — Deploys on any drone via Hardware Abstraction Layer (HAL) and containerized architecture
- **Human-in-the-Loop** — AI recommends actions, human approves. Autonomous only within ROE during disconnection

## Architecture

```
Camera → Vision Pipeline → Decision Engine → MAVLink → Flight Controller
  │         (YOLOv10)        (ROE-based)       ↕
  └→ VIO → Navigation EKF → CER Controller  Ground Station
         (GPS-denied nav)   (soft-kill defense)
```

See [docs/architecture/](docs/architecture/) for detailed design documents with Mermaid diagrams.

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv10 / YOLOv9 (ultralytics) |
| Pose Estimation | YOLO-Pose |
| Multi-Object Tracking | ByteTrack |
| Visual Odometry | ORB-SLAM3 / Basalt |
| Sensor Fusion | 15-state Extended Kalman Filter |
| Drone Communication | MAVLink (pymavlink) |
| AI Inference | TensorRT (Jetson) / ONNX Runtime (portable) |
| Containerization | Docker + NVIDIA Container Toolkit |
| Encryption | AES-256-GCM |

## Project Structure

```
aerialmind/
├── src/aerialmind/       # Main Python package
│   ├── core/             # Types, protocols, message bus, config
│   ├── hal/              # Hardware Abstraction Layer
│   ├── vision/           # Detection, tracking, behavior analysis
│   ├── navigation/       # VIO, EKF, CER, path planning
│   ├── decision/         # Threat assessment, ROE, autonomous actions
│   ├── comms/            # MAVLink, telemetry, encryption
│   └── modes/            # Military/Civil mode management
├── src/vio_cpp/          # C++ VIO and EKF (pybind11 bindings)
├── models/               # ONNX models + manifests
├── config/               # HAL, ROE, mode, and behavior configs
├── docker/               # Container definitions
├── tests/                # Unit, integration, simulation tests
├── tools/                # CLI utilities (model compiler, calibration)
└── docs/                 # Architecture and guide documentation
```

## Quick Start

```bash
# Clone
git clone https://github.com/omkarjoshi0304/aerialmind.git
cd aerialmind

# Install (development mode)
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"

# Run tests
pytest
```

## Hardware Support

| Platform | Accelerator | Status |
|---|---|---|
| NVIDIA Jetson Orin Nano | TensorRT (FP16) | Primary target |
| Raspberry Pi 5 + Hailo-8L | Hailo SDK | Planned |
| Any x86/ARM | ONNX Runtime (CPU) | Fallback |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and branching strategy.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
