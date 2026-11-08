"""
Project T253: Escalator Misuse & Unsafe Riding Behaviour Detection
Configuration Module
"""

from pathlib import Path
import os
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

# Model weights directory
MODELS_DIR = BASE_DIR / "models"

# Output directories
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUT_VIDEOS_DIR = OUTPUTS_DIR / "videos"
OUTPUT_ALERTS_DIR = OUTPUTS_DIR / "alerts"

# Reports directory
REPORTS_DIR = BASE_DIR / "reports"

# Hardware / Device configuration
def get_device() -> torch.device:
    """
    Dynamically select CUDA GPU if available and functional, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

DEVICE = get_device()
USE_CUDA = torch.cuda.is_available()

# Video processing configuration defaults
VIDEO_CONFIG = {
    "default_fps": 30,
    "input_resolution": (1920, 1080),
    "processing_resolution": (1280, 720),
    "fourcc": "mp4v",
    "batch_size": 1,
}

# Detection and Tracking parameters (scaffolding defaults)
DETECTION_CONFIG = {
    "model_name": "yolov8m.pt",  # Placeholder reference for Phase 2
    "confidence_threshold": 0.45,
    "iou_threshold": 0.50,
    "classes_of_interest": [0],  # Class 0 = person in COCO
}

TRACKING_CONFIG = {
    "tracker_type": "bytetrack",  # bytetrack or botsort
    "track_history_length": 60,
    "max_lost_frames": 30,
}

# Escalator behaviour definitions & alert thresholds
BEHAVIOUR_CONFIG = {
    "unusual_speed_multiplier": 1.75,
    "reverse_direction_threshold_frames": 15,
    "stationary_prolonged_seconds": 10.0,
    "crowding_density_threshold": 6,
    "climbing_handrail_confidence": 0.65,
}

# Alert generator settings
ALERT_CONFIG = {
    "log_format": "json",
    "alert_log_file": OUTPUT_ALERTS_DIR / "alerts_log.json",
    "enable_visual_overlay": True,
    "alert_cooldown_seconds": 5.0,
}
