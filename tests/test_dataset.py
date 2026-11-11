"""
Tests for Escalator Video Dataset and Preprocessing Pipeline
Project T253 — Escalator Misuse & Unsafe Riding Behaviour Detection
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from src.video_dataset import EscalatorVideoDataset

MANIFEST_PATH = Path("reports/split_manifest.csv")

def test_dataset_initialization():
    """Verify train, val, and test partitions initialize with valid sample counts."""
    assert MANIFEST_PATH.exists(), "split_manifest.csv must exist."
    
    train_ds = EscalatorVideoDataset(manifest_path=MANIFEST_PATH, split="train", num_frames=16)
    val_ds = EscalatorVideoDataset(manifest_path=MANIFEST_PATH, split="val", num_frames=16)
    test_ds = EscalatorVideoDataset(manifest_path=MANIFEST_PATH, split="test", num_frames=16)
    
    assert len(train_ds) == 3645
    assert len(val_ds) == 785
    assert len(test_ds) == 794
    assert len(train_ds) + len(val_ds) + len(test_ds) == 5224

def test_dataset_item_retrieval():
    """Verify loading a single video item yields expected tensor shape and label range."""
    val_ds = EscalatorVideoDataset(manifest_path=MANIFEST_PATH, split="val", num_frames=8, target_size=(112, 112))
    video_tensor, label, metadata = val_ds[0]
    
    # Expected shape: (Channels=3, Time=8, Height=112, Width=112)
    assert isinstance(video_tensor, torch.Tensor)
    assert video_tensor.shape == (3, 8, 112, 112)
    assert video_tensor.dtype == torch.float32
    assert 0 <= label <= 4
    assert "filename" in metadata
    assert "group_key" in metadata

def test_temporal_sampling_coverage():
    """Verify frame index sampler handles different video lengths."""
    val_ds = EscalatorVideoDataset(manifest_path=MANIFEST_PATH, split="val", num_frames=16)
    
    # Test case 1: Video longer than num_frames
    indices_long = val_ds._sample_frame_indices(100)
    assert len(indices_long) == 16
    assert np.all(np.diff(indices_long) >= 0)  # Monotonically increasing
    assert indices_long[-1] < 100
    
    # Test case 2: Video shorter than num_frames
    indices_short = val_ds._sample_frame_indices(5)
    assert len(indices_short) == 16
    assert indices_short[-1] < 5
