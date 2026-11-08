"""
Activity Classifier Module
Classifies individual person activities into standard or anomalous categories.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np


class ActivityType(Enum):
    """
    Standard escalator riding activity categories.
    """
    NORMAL_RIDING = "normal_riding"
    WALKING_FORWARD = "walking_forward"
    RUNNING = "running"
    REVERSE_WALKING = "reverse_walking"
    SITTING_ON_STEPS = "sitting_on_steps"
    HANDRAIL_CLIMBING = "handrail_climbing"
    FALLEN_PERSON = "fallen_person"
    LOITERING_ENTRY_EXIT = "loitering_entry_exit"
    UNKNOWN = "unknown"


class ActivityClassifier:
    """
    Performs classification of temporal motion and posture patterns into ActivityTypes.
    """

    def __init__(self, confidence_threshold: float = 0.60) -> None:
        """
        Initialize the ActivityClassifier.

        Args:
            confidence_threshold: Minimum confidence score to output a definitive activity label.
        """
        self.confidence_threshold = confidence_threshold

    def classify(
        self,
        track_history: List[Dict[str, Any]],
        pose_keypoints: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Classifies the activity of an individual based on track trajectory and pose.

        Args:
            track_history: Chronological list of position coordinates for a person.
            pose_keypoints: Optional skeletal keypoints from pose estimation.

        Returns:
            Dict containing activity label, confidence score, and timestamp.
        """
        # Placeholder for Phase 5: Activity classification model
        raise NotImplementedError("Activity classification will be implemented in Phase 5.")
