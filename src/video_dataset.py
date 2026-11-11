"""
Escalator Video Dataset Module
Project T253 — Escalator Misuse & Unsafe Riding Behaviour Detection

Provides high-performance, leakage-free PyTorch Dataset implementations
for video activity classification from CCTV footage.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EscalatorVideoDataset(Dataset):
    """
    PyTorch Dataset for loading video clips for temporal action recognition.
    Performs on-the-fly video decoding and uniform temporal sampling.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        split: str = "train",
        num_frames: int = 16,
        target_size: Tuple[int, int] = (224, 224),
        temporal_jitter: bool = False,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Initialize the EscalatorVideoDataset.

        Args:
            manifest_path: Path to split_manifest.csv.
            split: One of 'train', 'val', 'test'.
            num_frames: Number of frames to sample per video clip (default: 16).
            target_size: (height, width) spatial dimensions (default: 224x224).
            temporal_jitter: If True, randomly offsets sampling within segments (training only).
            transform: Optional torchvision or albumentations transform function.
        """
        self.manifest_path = Path(manifest_path)
        self.split = split.lower()
        self.num_frames = num_frames
        self.target_size = target_size
        self.temporal_jitter = temporal_jitter and (self.split == "train")
        self.transform = transform

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        df = pd.read_csv(self.manifest_path)
        if "split" not in df.columns:
            raise ValueError("Manifest must contain 'split' column.")

        self.samples = df[df["split"] == self.split].reset_index(drop=True)
        if len(self.samples) == 0:
            raise ValueError(f"No samples found for split '{self.split}'.")

        # Standard ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_frame_indices(self, total_frames: int) -> np.ndarray:
        """
        Calculates frame indices using uniform segment sampling.
        """
        if total_frames <= self.num_frames:
            # Video is shorter than required frames: repeat frames with padding
            indices = np.linspace(0, max(0, total_frames - 1), self.num_frames, dtype=int)
            return indices

        segment_length = total_frames / float(self.num_frames)
        indices = []
        for i in range(self.num_frames):
            start = int(i * segment_length)
            end = int((i + 1) * segment_length)
            if self.temporal_jitter and end > start:
                idx = np.random.randint(start, end)
            else:
                idx = int(start + segment_length / 2.0)
            indices.append(min(idx, total_frames - 1))
        return np.array(indices, dtype=int)

    def _load_video_frames(self, video_path: str) -> np.ndarray:
        """
        Opens video using OpenCV and reads the specified sample frames.
        Returns array of shape (T, H, W, C) in RGB format.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # Return zero tensor fallback if file cannot be opened
            return np.zeros((self.num_frames, self.target_size[0], self.target_size[1], 3), dtype=np.uint8)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = self.num_frames

        frame_indices = self._sample_frame_indices(total_frames)
        frames = []
        current_frame_idx = 0

        # Read frames sequentially or jump to index
        for target_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                # Fallback to black frame if decode glitch occurs
                frame = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.target_size[1], self.target_size[0]))
            frames.append(frame)

        cap.release()
        return np.array(frames, dtype=np.uint8)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Union[str, int, float]]]:
        """
        Retrieves a sample from the dataset.

        Returns:
            Tuple containing:
                - video_tensor: torch.FloatTensor of shape (C, T, H, W)
                - label: int class_id (0 to 4)
                - metadata: dict of sample details
        """
        row = self.samples.iloc[idx]
        video_path = row["file_path"]
        class_id = int(row["class_id"])

        # Load raw RGB frames (T, H, W, C)
        frames_np = self._load_video_frames(video_path)

        # Apply basic training-time horizontal flip if enabled
        if self.split == "train" and np.random.rand() > 0.5:
            frames_np = np.flip(frames_np, axis=2).copy()

        # Convert to float32 [0.0, 1.0] and normalize
        frames_float = frames_np.astype(np.float32) / 255.0
        normalized_frames = (frames_float - self.mean) / self.std

        # Permute from (T, H, W, C) to (C, T, H, W) standard PyTorch video format
        video_tensor = torch.from_numpy(normalized_frames).permute(3, 0, 1, 2)

        metadata = {
            "filename": row["filename"],
            "group_key": row["group_key"],
            "class_name": row["t253_target_class"],
            "duration": float(row["duration_seconds"]),
        }

        return video_tensor, class_id, metadata
