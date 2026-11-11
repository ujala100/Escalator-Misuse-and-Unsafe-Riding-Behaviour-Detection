"""
Project T253: Escalator Misuse & Unsafe Riding Behaviour Detection
Environment and Setup Test Suite
"""

import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np


def test_python_version():
    """Verify that Python version meets or exceeds 3.10."""
    assert sys.version_info >= (3, 10), f"Python version {sys.version} is below 3.10"


def test_opencv_installation():
    """Verify OpenCV installation and basic matrix operations."""
    import cv2
    assert cv2.__version__ is not None
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert gray.shape == (100, 100)


def test_opencv_video_io():
    """Verify OpenCV video encoding and decoding capabilities."""
    import cv2
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_video = str(Path(tmp_dir) / "test.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(test_video, fourcc, 25.0, (64, 64))
        assert writer.isOpened(), "VideoWriter could not open file."
        frame = np.ones((64, 64, 3), dtype=np.uint8) * 128
        writer.write(frame)
        writer.release()

        cap = cv2.VideoCapture(test_video)
        assert cap.isOpened(), "VideoCapture could not open file."
        ret, read_frame = cap.read()
        cap.release()
        assert ret is True
        assert read_frame.shape == (64, 64, 3)


def test_numpy_and_pandas():
    """Verify NumPy and Pandas installation and basic operations."""
    import numpy as np
    import pandas as pd

    arr = np.array([1, 2, 3, 4, 5])
    assert arr.mean() == 3.0

    df = pd.DataFrame({"id": [1, 2], "score": [0.95, 0.88]})
    assert len(df) == 2
    assert "score" in df.columns


def test_scipy_and_matplotlib():
    """Verify SciPy and Matplotlib installation."""
    from scipy.spatial.distance import euclidean
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for headless test
    import matplotlib.pyplot as plt

    dist = euclidean([0, 0], [3, 4])
    assert dist == 5.0

    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    plt.close(fig)


def test_pillow_yaml_tqdm():
    """Verify Pillow, PyYAML, and tqdm utilities."""
    from PIL import Image
    import yaml
    import tqdm

    img = Image.new("RGB", (32, 32), color="red")
    assert img.size == (32, 32)

    data = {"project": "T253", "phase": 1}
    yaml_str = yaml.dump(data)
    loaded = yaml.safe_load(yaml_str)
    assert loaded["project"] == "T253"

    pbar = tqdm.tqdm(total=5, disable=True)
    pbar.update(1)
    pbar.close()


def test_pytorch_and_torchvision():
    """Verify PyTorch and Torchvision tensor calculations on available device."""
    import torch
    import torchvision

    assert torch.__version__ is not None
    assert torchvision.__version__ is not None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t1 = torch.ones((3, 3), device=device)
    t2 = torch.ones((3, 3), device=device) * 2
    t3 = t1 + t2
    assert t3.sum().item() == 27.0


def test_ultralytics_import():
    """Verify Ultralytics YOLO module import without triggering downloads."""
    from ultralytics import YOLO
    assert YOLO is not None


def test_project_structure():
    """Verify that all required project directories exist."""
    import config

    assert config.BASE_DIR.exists()
    assert config.DATA_DIR.exists()
    assert config.RAW_DATA_DIR.exists()
    assert config.PROCESSED_DATA_DIR.exists()
    assert config.SAMPLE_DATA_DIR.exists()
    assert config.MODELS_DIR.exists()
    assert config.OUTPUTS_DIR.exists()
    assert config.OUTPUT_VIDEOS_DIR.exists()
    assert config.OUTPUT_ALERTS_DIR.exists()
    assert config.REPORTS_DIR.exists()


def test_src_modules_import():
    """Verify that all core src modules import cleanly."""
    from src import (
        VideoProcessor,
        PersonDetector,
        PersonTracker,
        BehaviourDetector,
        ActivityClassifier,
        AlertGenerator,
    )
    assert VideoProcessor is not None
    assert PersonDetector is not None
    assert PersonTracker is not None
    assert BehaviourDetector is not None
    assert ActivityClassifier is not None
    assert AlertGenerator is not None
