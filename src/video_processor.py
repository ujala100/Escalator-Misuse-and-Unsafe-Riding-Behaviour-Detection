"""
Video Processor Module
Handles video ingestion, frame extraction, pre-processing, and video stream output.
"""

from pathlib import Path
from typing import Generator, Optional, Tuple, Union
import cv2
import numpy as np


class VideoProcessor:
    """
    Manages reading from CCTV video feeds/files and writing processed video streams.
    """

    def __init__(
        self,
        source: Union[str, Path, int],
        target_resolution: Optional[Tuple[int, int]] = None,
        target_fps: Optional[int] = None,
    ) -> None:
        """
        Initialize the VideoProcessor.

        Args:
            source: Path to video file or RTSP stream URL / camera index.
            target_resolution: Optional (width, height) to resize frames.
            target_fps: Optional target FPS for processing.
        """
        self.source = str(source)
        self.target_resolution = target_resolution
        self.target_fps = target_fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.total_frames: int = 0
        self.fps: float = 0.0

    def open(self) -> bool:
        """
        Opens the video capture stream.

        Returns:
            bool: True if capture stream was successfully opened, False otherwise.
        """
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            return False

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        return True

    def stream_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Yields frames sequentially from the open video stream.

        Yields:
            Tuple[int, np.ndarray]: Frame index and decoded BGR image frame.
        """
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Video source is not open. Call open() first.")

        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break

            if self.target_resolution is not None:
                frame = cv2.resize(frame, self.target_resolution)

            yield frame_idx, frame
            frame_idx += 1

    def release(self) -> None:
        """
        Releases video capture and video writer resources.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
