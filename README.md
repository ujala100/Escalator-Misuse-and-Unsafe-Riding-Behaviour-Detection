# Project T253 – Escalator Misuse & Unsafe Riding Behaviour Detection

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green.svg)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-blueviolet.svg)
![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)

---

## 1. Project Introduction

Public transit stations, shopping malls, airports, and commercial centers rely heavily on escalators to move large volumes of people efficiently. However, escalators represent one of the most frequent sites of public mechanical accidents. Unsafe passenger behaviours—such as riding in the opposite direction, running on moving treads, leaning over or riding handrails, sitting on steps, carrying excessive or improper luggage, and entrance/exit crowding—result in thousands of preventable injuries and emergency stoppages each year.

**Project T253** provides an automated, real-time Computer Vision system that analyses escalator CCTV camera feeds. By detecting people, tracking their spatio-temporal movements across defined escalator zones, classifying their riding activities, and flagging anomalous and hazardous behaviours, the system produces actionable, timestamped alerts and visual overlays for facility safety operators.

---

## 2. Project Objectives

1. **Automated Person Detection:** Robustly locate individuals entering, riding, and exiting the escalator corridor under varying lighting, densities, and camera angles.
2. **Multi-Object Tracking:** Assign persistent unique track IDs to passengers and construct movement trajectories across successive video frames.
3. **Escalator Spatial Zone Mapping:** Define regions of interest (Entrance Landings, Incline Treadway, Handrails, and Exit Landings) and escalator travel direction (Upward / Downward).
4. **Unsafe Behaviour & Misuse Detection:** Detect dangerous behaviours including:
   - Walking or running against the direction of travel (reverse walking).
   - Speed anomalies (running or rushing down steps).
   - Sitting, lying, or falling on steps.
   - Handrail misuse (leaning over, sitting, or climbing).
   - Obstruction or loitering at critical landing combs.
   - Overcrowding and sudden group turbulence.
5. **Activity Classification:** Categorize passenger actions into standard versus hazardous categories.
6. **Alert Dispatching & Logging:** Generate timestamped JSON/JSONL event logs, visual warning bounding boxes, and operator notifications.

---

## 3. Technology Stack

| Category | Technology / Library | Version / Specification | Role in Project |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.10.x (64-bit) | Primary development runtime |
| **Deep Learning Framework** | PyTorch / Torchvision | 2.6.0+cu124 / 0.21.0+cu124 | Neural network compute and GPU acceleration |
| **Object Detection & Pose** | Ultralytics YOLO | 8.x | Person detection, bounding box prediction, pose keypoints |
| **Computer Vision & Video** | OpenCV (`opencv-python`) | 5.0.0 | Video decoding, frame processing, FFMPEG I/O, overlays |
| **Array & Numerical Compute**| NumPy | 2.2.x | High-performance matrix and vector calculations |
| **Spatial & Analytics** | SciPy | 1.15.x | Trajectory distance calculation, spatial clustering |
| **Data Handling** | Pandas | 2.3.x | Tabular alert logging and performance metrics analysis |
| **Imaging & Visualization** | Pillow, Matplotlib | 12.3.x, 3.10.x | Image manipulation, trajectory graphing, diagnostic plots |
| **Testing & Quality** | Pytest | 9.2.x | Automated test suite and environment verification |
| **Hardware Target** | NVIDIA GPU / CUDA | CUDA 12.4 / Driver >= 566.36 | Accelerated model inference and tensor computations |

---

## 4. Directory Structure

```
Escalator-Misuse-and-Unsafe-Riding-Behaviour-Detection/
│
├── data/                               # Dataset repository
│   ├── raw/                            # Unprocessed CCTV video files and raw clips
│   ├── processed/                      # Extracted frame sequences and preprocessed data
│   └── sample/                         # Lightweight sample clips for development testing
│
├── models/                             # Trained model checkpoints, YOLO weights (.pt, .onnx)
│
├── src/                                # Core application source code
│   ├── __init__.py                     # Package export and version definition
│   ├── video_processor.py              # Video stream ingestion, decoding, frame generator
│   ├── person_detector.py              # Person detection module interface
│   ├── person_tracker.py               # Spatio-temporal multi-object tracking (ByteTrack)
│   ├── behaviour_detector.py           # Escalator rule analysis and anomaly detection
│   ├── activity_classifier.py          # Riding activity classification engine
│   └── alert_generator.py              # Timestamped alert logger and visual overlay engine
│
├── tests/                              # Automated test suite
│   └── test_environment.py             # Environment, import, and subsystem verification tests
│
├── reports/                            # Generated performance benchmarks and safety reports
│
├── outputs/                            # Model inference outputs
│   ├── videos/                         # Annotated video streams with visual bounding boxes
│   └── alerts/                         # Structured JSON/JSONL event logs
│
├── scripts/                            # Utility and maintenance scripts
│   └── check_environment.py            # Comprehensive environment verification script
│
├── .gitignore                          # Comprehensive Git exclusion rules
├── requirements.txt                    # Project dependencies
├── README.md                           # Documentation and setup guide
└── config.py                           # Central configuration constants and paths
```

---

## 5. Windows 11 Setup Instructions

### Prerequisites
- Windows 11 (64-bit)
- Python 3.10 (Recommended fallback when Python 3.11 is not globally installed)
- PowerShell (or Windows Terminal)
- Git for Windows

### Step 1: Open PowerShell in the Project Directory
Navigate to the root directory of the repository:
```powershell
cd "e:\Escalator Misuse Detection\Escalator-Misuse-and-Unsafe-Riding-Behaviour-Detection"
```

### Step 2: Virtual Environment Configuration
If the `.venv` virtual environment is not already created, run:
```powershell
py -3.10 -m venv .venv
```

Activate the virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(Note: If PowerShell displays an `Execution_Policies` restriction, enable local script execution via `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*.

Upgrade pip inside `.venv`:
```powershell
python -m pip install --upgrade pip
```

---

## 6. Dependency Installation

### Step 1: Install PyTorch with CUDA Support
For Windows with an NVIDIA GPU and driver 566.36+, install PyTorch built with CUDA 12.4:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
*(If developing purely on CPU without an NVIDIA GPU, install using standard pip: `pip install torch torchvision`)*.

### Step 2: Install Remaining Computer Vision & Core Dependencies
```powershell
pip install -r requirements.txt
```

---

## 7. GPU Verification and Troubleshooting

To test whether PyTorch can detect your NVIDIA GPU:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Troubleshooting NVIDIA GPU on Windows Laptops (ASUS / Advanced Optimus / Eco Mode):
If `torch.cuda.is_available()` returns `False` or `nvidia-smi` reports insufficient permissions:
1. **GPU Power Gate (Eco Mode):** Many dual-GPU laptops (AMD Radeon + NVIDIA RTX 2050/3050/4060) feature hardware MUX switches. In ASUS Armoury Crate / G-Helper or Windows Power settings, ensure the GPU Mode is set to **Standard** or **Optimized**, not **Eco Mode** (which detaches the dGPU PCIe device, placing it in a phantom state `CM_PROB_PHANTOM`).
2. **NVIDIA Display Container Service:** Open `services.msc` as Administrator, find `NVIDIA Display Container LS`, and ensure its startup type is Automatic and the service is Running.
3. **Automatic Fallback:** The codebase (`config.py` and `scripts/check_environment.py`) is designed with resilient hardware detection. If CUDA is not engaged, the system operates in **CPU Mode** seamlessly. Once the discrete GPU is re-engaged, PyTorch will immediately utilize CUDA acceleration without requiring reinstallation.

---

## 8. Environment Verification Commands

To verify that the environment is fully operational:

### 1. Run the Verification Script:
```powershell
python scripts/check_environment.py
```
This script validates:
- Python version (>= 3.10)
- OpenCV installation and version
- NumPy, Pandas, Ultralytics, PyTorch, Torchvision installations
- CUDA availability and NVIDIA GPU identification
- OpenCV VideoCapture & VideoWriter pipeline (mp4v codec test)

### 2. Run the Pytest Test Suite:
```powershell
pytest -v
```

All 10 test cases should pass successfully.

---

## 9. Future Implementation Stages

The project is structured in sequential phases:

```
+-----------------------------------------------------------------------------------+
| Phase 1: Environment Setup & Project Scaffolding (Current Phase)                 |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| Phase 2: Video Ingestion & Person Detection Pipeline                              |
| - Implement VideoProcessor stream reading with buffer management                  |
| - Integrate YOLOv8 person detection with confidence calibration                   |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| Phase 3: Multi-Object Tracking & Motion Trajectories                              |
| - Integrate ByteTrack / BoT-SORT for persistent passenger IDs                     |
| - Compute velocity vectors, centroid histories, and trajectory smoothing          |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| Phase 4: Escalator Zone Mapping & Unsafe Behaviour Detection                      |
| - Interactive polygon mapping for escalator entry, exit, treads, and handrails    |
| - Implement algorithmic rules for reverse walking, running, falls, and loitering  |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| Phase 5: Activity Classification & Alert Dispatcher                               |
| - Temporal posture and movement classification                                    |
| - Generate JSON/JSONL event logs and render real-time bounding box alert overlays |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| Phase 6: System Evaluation, Dashboard & Deployment                                |
| - Compute precision, recall, and false-alarm rates across CCTV test benchmarks    |
| - Package executable demo and generate comprehensive technical report             |
+-----------------------------------------------------------------------------------+
```
