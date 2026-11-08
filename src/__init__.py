"""
Project T253: Escalator Misuse & Unsafe Riding Behaviour Detection
Core Source Package
"""

__version__ = "0.1.0"
__author__ = "Project T253 Team"

from .video_processor import VideoProcessor
from .person_detector import PersonDetector
from .person_tracker import PersonTracker
from .behaviour_detector import BehaviourDetector
from .activity_classifier import ActivityClassifier
from .alert_generator import AlertGenerator

__all__ = [
    "VideoProcessor",
    "PersonDetector",
    "PersonTracker",
    "BehaviourDetector",
    "ActivityClassifier",
    "AlertGenerator",
]
