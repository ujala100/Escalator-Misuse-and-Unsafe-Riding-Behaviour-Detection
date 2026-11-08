#!/usr/bin/env python3
"""
Environment Verification Script
Project T253: Escalator Misuse & Unsafe Riding Behaviour Detection

Verifies all essential computer vision, deep learning, and data processing dependencies,
hardware acceleration (CUDA/NVIDIA GPU), and OpenCV video processing capabilities.
"""

import sys
import os
import tempfile
from typing import Dict, Tuple, Any

# Minimum Python version requirement
MIN_PYTHON_VERSION = (3, 10)


def check_python() -> Tuple[bool, str]:
    version = sys.version_info
    ver_str = f"{version.major}.{version.minor}.{version.micro}"
    if version >= MIN_PYTHON_VERSION:
        return True, f"Python {ver_str} (>= 3.10 required, detected {ver_str})"
    return False, f"Python {ver_str} is below minimum requirement 3.10"


def check_opencv() -> Tuple[bool, str]:
    try:
        import cv2
        return True, f"OpenCV {cv2.__version__}"
    except ImportError as e:
        return False, f"Failed to import OpenCV: {e}"


def check_numpy() -> Tuple[bool, str]:
    try:
        import numpy as np
        return True, f"NumPy {np.__version__}"
    except ImportError as e:
        return False, f"Failed to import NumPy: {e}"


def check_pandas() -> Tuple[bool, str]:
    try:
        import pandas as pd
        return True, f"Pandas {pd.__version__}"
    except ImportError as e:
        return False, f"Failed to import Pandas: {e}"


def check_ultralytics() -> Tuple[bool, str]:
    try:
        import ultralytics
        return True, f"Ultralytics {ultralytics.__version__}"
    except ImportError as e:
        return False, f"Failed to import Ultralytics: {e}"


def check_pytorch() -> Tuple[bool, str]:
    try:
        import torch
        return True, f"PyTorch {torch.__version__}"
    except ImportError as e:
        return False, f"Failed to import PyTorch: {e}"


def check_torchvision() -> Tuple[bool, str]:
    try:
        import torchvision
        return True, f"Torchvision {torchvision.__version__}"
    except ImportError as e:
        return False, f"Failed to import Torchvision: {e}"


def check_cuda_and_gpu() -> Tuple[bool, str, str]:
    """
    Checks CUDA availability and retrieves GPU device name if present.
    Does not crash if CUDA is unavailable.
    """
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            device_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            cuda_ver = torch.version.cuda
            return True, f"CUDA Available (v{cuda_ver}, {device_count} device(s) detected)", gpu_name
        else:
            cuda_build = getattr(torch.version, "cuda", "N/A")
            return False, f"CUDA Unavailable (PyTorch built with CUDA {cuda_build}; running on CPU)", "N/A (CPU Mode)"
    except Exception as e:
        return False, f"Error inspecting CUDA: {e}", "N/A"


def check_opencv_video_processing() -> Tuple[bool, str]:
    """
    Verifies OpenCV VideoWriter and VideoCapture pipeline by writing and reading a synthetic frame.
    """
    try:
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_video_path = os.path.join(tmp_dir, "test_check_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            width, height = 128, 128
            fps = 30.0

            # Write a test frame
            writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                return False, "OpenCV VideoWriter failed to open MP4 container with mp4v codec."

            test_frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(test_frame, "T253", (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            writer.write(test_frame)
            writer.release()

            # Read back the test frame
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                return False, "OpenCV VideoCapture failed to read back test video."

            ret, read_frame = cap.read()
            cap.release()

            if not ret or read_frame is None or read_frame.shape != (height, width, 3):
                return False, "OpenCV VideoCapture failed to read frame with expected dimensions."

        return True, "OpenCV VideoCapture & VideoWriter verified (mp4v codec operational)"
    except Exception as e:
        return False, f"OpenCV video processing test failed: {e}"


def run_all_checks() -> bool:
    print("=" * 80)
    print("Project T253 – Escalator Misuse & Unsafe Riding Behaviour Detection")
    print("Environment Verification Report")
    print("=" * 80)

    checks = []

    # 1. Python version
    py_pass, py_msg = check_python()
    checks.append(("1. Python Version", py_pass, py_msg, False))

    # 2. OpenCV installation
    cv_pass, cv_msg = check_opencv()
    checks.append(("2. OpenCV Installation", cv_pass, cv_msg, False))

    # 3. NumPy installation
    np_pass, np_msg = check_numpy()
    checks.append(("3. NumPy Installation", np_pass, np_msg, False))

    # 4. Pandas installation
    pd_pass, pd_msg = check_pandas()
    checks.append(("4. Pandas Installation", pd_pass, pd_msg, False))

    # 5. Ultralytics installation
    ul_pass, ul_msg = check_ultralytics()
    checks.append(("5. Ultralytics Installation", ul_pass, ul_msg, False))

    # 6. PyTorch installation
    th_pass, th_msg = check_pytorch()
    checks.append(("6. PyTorch Installation", th_pass, th_msg, False))

    # 7. Torchvision installation
    tv_pass, tv_msg = check_torchvision()
    checks.append(("7. Torchvision Installation", tv_pass, tv_msg, False))

    # 8. CUDA Availability & 9. GPU Name
    cuda_pass, cuda_msg, gpu_name = check_cuda_and_gpu()
    # CUDA/GPU is optional hardware acceleration (fallback to CPU is permitted and handled gracefully)
    checks.append(("8. CUDA Availability", cuda_pass, cuda_msg, True))
    checks.append(("9. NVIDIA GPU Name", cuda_pass, f"Device: {gpu_name}", True))

    # 10. OpenCV Video Processing
    vid_pass, vid_msg = check_opencv_video_processing()
    checks.append(("10. OpenCV Video I/O", vid_pass, vid_msg, False))

    all_core_passed = True

    for name, passed, msg, is_optional in checks:
        if passed:
            status_tag = "[PASS]"
        elif is_optional:
            status_tag = "[INFO/FALLBACK]"
        else:
            status_tag = "[FAIL]"
            all_core_passed = False

        print(f"{status_tag:<16} {name:<28} : {msg}")

    print("=" * 80)
    if all_core_passed:
        if cuda_pass:
            print("STATUS: ENVIRONMENT FULLY READY (GPU ACCELERATION ACTIVE)")
        else:
            print("STATUS: CORE ENVIRONMENT READY (CPU MODE ACTIVE / GPU FALLBACK DOCUMENTED)")
        print("=" * 80)
        return True
    else:
        print("STATUS: ENVIRONMENT CHECK FAILED (CRITICAL DEPENDENCIES MISSING)")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
