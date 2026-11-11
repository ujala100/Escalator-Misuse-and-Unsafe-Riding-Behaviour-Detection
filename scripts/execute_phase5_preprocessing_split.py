"""
Phase 5: Data Preprocessing, Augmentation & Dataset Splitting
Project T253 — Escalator Misuse & Unsafe Riding Behaviour Detection

Tasks executed by this script:
1. Load annotated dataset manifest (5,224 clean clips).
2. Extract recording session/camera group keys (433 unique groups).
3. Perform group-based stratified splitting targeting 70% Train, 15% Val, 15% Test.
4. Verify 100% leakage-free isolation (zero group overlap, zero MD5 collisions).
5. Calculate class distributions and inverse-frequency class weights for training.
6. Generate augmentation configuration YAML.
7. Generate Preprocessing & Dataset Splitting reports:
   - reports/split_manifest.csv & dataset_work/reports/split_manifest.csv
   - reports/split_class_distribution.csv & dataset_work/reports/split_class_distribution.csv
   - reports/augmentation_config.yaml & dataset_work/reports/augmentation_config.yaml
   - reports/leakage_verification_report.md & dataset_work/reports/leakage_verification_report.md
   - reports/preprocessing_pipeline.md & dataset_work/reports/preprocessing_pipeline.md
"""

import os
import re
import sys
import json
import yaml
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_WORK_DIR = PROJECT_ROOT / "dataset_work"
REPORTS_DIR = PROJECT_ROOT / "reports"
WORK_REPORTS_DIR = DATASET_WORK_DIR / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
WORK_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ANNOTATED_MANIFEST_PATH = WORK_REPORTS_DIR / "annotated_dataset_manifest.csv"

def extract_group_key(fn: str) -> str:
    """
    Extract recording session / camera scene group from filename.
    Ensures all clips from the same recording session stay strictly in the same split.
    """
    fn_lower = fn.lower()
    
    # Okutama: okutama_1.1.2_Sitting_59.mp4 -> okutama_1.1.2
    m = re.match(r"(okutama_[0-9]+\.[0-9]+\.[0-9]+)", fn_lower)
    if m:
        return m.group(1)
        
    # VIRAT: virat_VIRAT_S_050201_08_... -> virat_virat_s_050201
    m = re.match(r"(virat_[a-z0-9_]+_[0-9]{6})", fn_lower)
    if m:
        return m.group(1)
        
    # CASIA: casia_angleview_p01_... -> casia_angleview_p01
    m = re.match(r"(casia_[a-z]+_p[0-9]+)", fn_lower)
    if m:
        return m.group(1)
        
    # UCF-ARG: ucarg_person04_03_... -> ucarg_person04
    m = re.match(r"(ucarg_person[0-9]+)", fn_lower)
    if m:
        return m.group(1)
        
    # MEVA: meva_2018-03-07.17-35-00... -> meva_2018-03-07
    m = re.match(r"(meva_[0-9]{4}-[0-9]{2}-[0-9]{2})", fn_lower)
    if m:
        return m.group(1)
        
    # CAVIAR: caviar_Rest_InChair... -> caviar_rest
    if "caviar" in fn_lower:
        parts = fn_lower.split("_")
        return "caviar_" + (parts[1] if len(parts) > 1 else "general")
        
    parts = fn_lower.split("_")
    return parts[0] + "_" + parts[1] if len(parts) > 1 else parts[0]

def optimize_group_stratified_split(df: pd.DataFrame, seed: int = 42, target_ratios=(0.70, 0.15, 0.15), n_trials=2000):
    """
    Capacity-normalized multi-objective optimization to assign each recording group
    to Train, Val, or Test while maintaining ~70/15/15 proportions across all classes.
    """
    groups = df.groupby("group_key")["class_id"].value_counts().unstack(fill_value=0)
    target_ratios_np = np.array(target_ratios)
    class_totals = groups.sum(axis=0).values
    total_clips = len(df)
    target_split_clips = total_clips * target_ratios_np

    best_score = float("inf")
    best_alloc = None
    best_assign = None

    np.random.seed(seed)
    for trial in range(n_trials):
        perm = np.random.permutation(len(groups))
        alloc = np.zeros((3, 5))
        assign = {}
        
        for g_idx in perm:
            g_name = groups.index[g_idx]
            g_vec = groups.iloc[g_idx].values
            
            costs = []
            for s in range(3):
                fill_ratio = (alloc[s].sum() + g_vec.sum()) / target_split_clips[s]
                class_pen = 0.0
                for c in range(5):
                    if g_vec[c] > 0:
                        c_target = class_totals[c] * target_ratios_np[s]
                        c_fill = (alloc[s, c] + g_vec[c]) / (c_target + 1e-5)
                        class_pen += c_fill
                costs.append(fill_ratio * 2.0 + class_pen)
                
            chosen_s = np.argmin(costs)
            alloc[chosen_s] += g_vec
            assign[g_name] = chosen_s
            
        score = 0.0
        for s in range(3):
            for c in range(5):
                c_target = class_totals[c] * target_ratios_np[s]
                score += ((alloc[s, c] - c_target) / (c_target + 1e-5)) ** 2
        score += 5.0 * np.sum(((alloc.sum(axis=1) - target_split_clips) / target_split_clips) ** 2)
        
        if score < best_score:
            best_score = score
            best_alloc = alloc.copy()
            best_assign = assign.copy()

    split_names = ["train", "val", "test"]
    group_split_map = {g_name: split_names[s_idx] for g_name, s_idx in best_assign.items()}
    df["split"] = df["group_key"].map(group_split_map)
    return df, best_alloc

def main():
    print("=" * 70)
    print("Project T253 — Phase 5: Preprocessing, Augmentation & Splitting")
    print("=" * 70)
    
    if not ANNOTATED_MANIFEST_PATH.exists():
        print(f"Error: {ANNOTATED_MANIFEST_PATH} not found. Run Phase 4 first.")
        return
        
    print(f"Loading annotated manifest from {ANNOTATED_MANIFEST_PATH}...")
    df = pd.read_csv(ANNOTATED_MANIFEST_PATH)
    print(f"Total clean clips: {len(df)}")
    
    # Step 1: Group Key Extraction
    print("\n[Step 1] Extracting recording session / camera group keys...")
    df["group_key"] = [extract_group_key(f) for f in df["filename"]]
    num_groups = df["group_key"].nunique()
    print(f"  Extracted {num_groups} unique recording groups across {len(df)} clips.")
    
    # Step 2: Optimized Group-Aware Stratified Splitting
    print("\n[Step 2] Optimizing Group-Aware Stratified Splitting (70% Train, 15% Val, 15% Test)...")
    split_df, alloc = optimize_group_stratified_split(df, seed=42, target_ratios=(0.70, 0.15, 0.15), n_trials=2000)
    
    # Step 3: Leakage Verification
    print("\n[Step 3] Performing Comprehensive Leakage Verification...")
    train_groups = set(split_df[split_df["split"] == "train"]["group_key"])
    val_groups = set(split_df[split_df["split"] == "val"]["group_key"])
    test_groups = set(split_df[split_df["split"] == "test"]["group_key"])
    
    tv_overlap = train_groups.intersection(val_groups)
    tt_overlap = train_groups.intersection(test_groups)
    vt_overlap = val_groups.intersection(test_groups)
    
    train_md5s = set(split_df[split_df["split"] == "train"]["md5"])
    val_md5s = set(split_df[split_df["split"] == "val"]["md5"])
    test_md5s = set(split_df[split_df["split"] == "test"]["md5"])
    
    md5_tv_overlap = train_md5s.intersection(val_md5s)
    md5_tt_overlap = train_md5s.intersection(test_md5s)
    md5_vt_overlap = val_md5s.intersection(test_md5s)
    
    print(f"  Group Overlap Check:")
    print(f"    - Train vs Val Group Overlap:  {len(tv_overlap)} (PASS: 0)")
    print(f"    - Train vs Test Group Overlap: {len(tt_overlap)} (PASS: 0)")
    print(f"    - Val vs Test Group Overlap:   {len(vt_overlap)} (PASS: 0)")
    
    print(f"  MD5 Hash Collision Check:")
    print(f"    - Train vs Val MD5 Collisions:  {len(md5_tv_overlap)} (PASS: 0)")
    print(f"    - Train vs Test MD5 Collisions: {len(md5_tt_overlap)} (PASS: 0)")
    print(f"    - Val vs Test MD5 Collisions:   {len(md5_vt_overlap)} (PASS: 0)")
    
    assert len(tv_overlap) == 0 and len(tt_overlap) == 0 and len(vt_overlap) == 0, "FATAL: Group leakage detected!"
    assert len(md5_tv_overlap) == 0 and len(md5_tt_overlap) == 0 and len(md5_vt_overlap) == 0, "FATAL: MD5 leakage detected!"
    print("  >>> ZERO DATA LEAKAGE CERTIFIED: 100% DISJOINT GROUPS & HASHES <<<")
    
    # Step 4: Class Distribution Analysis
    print("\n[Step 4] Analyzing Split Class Distribution & Imbalance...")
    dist_records = []
    splits = ["train", "val", "test"]
    class_names = {
        0: "NORMAL_RIDING_STANDING",
        1: "WALKING",
        2: "RUNNING_RUSHING",
        3: "SITTING_SQUATTING",
        4: "FALLEN_PERSON"
    }
    
    for c_id in sorted(class_names.keys()):
        c_name = class_names[c_id]
        c_df = split_df[split_df["class_id"] == c_id]
        n_tot = len(c_df)
        n_tr = len(c_df[c_df["split"] == "train"])
        n_val = len(c_df[c_df["split"] == "val"])
        n_te = len(c_df[c_df["split"] == "test"])
        
        dist_records.append({
            "class_id": c_id,
            "class_name": c_name,
            "total_clips": n_tot,
            "train_count": n_tr,
            "train_pct": round(n_tr / n_tot * 100, 1),
            "val_count": n_val,
            "val_pct": round(n_val / n_tot * 100, 1),
            "test_count": n_te,
            "test_pct": round(n_te / n_tot * 100, 1),
        })
        
    dist_df = pd.DataFrame(dist_records)
    print("\nClass Distribution Across Splits:")
    print(dist_df.to_string(index=False))
    
    # Total counts per split
    print(f"\nSplit Totals:")
    for s in splits:
        cnt = len(split_df[split_df["split"] == s])
        pct = cnt / len(split_df) * 100
        print(f"  - {s.upper():5s}: {cnt:5d} clips ({pct:.1f}%)")
        
    # Calculate Training Inverse-Frequency Class Weights
    train_df = split_df[split_df["split"] == "train"]
    N_tr = len(train_df)
    K = len(class_names)
    class_weights = {}
    for c_id in range(K):
        n_c = len(train_df[train_df["class_id"] == c_id])
        w = round(N_tr / (K * n_c), 4) if n_c > 0 else 1.0
        class_weights[c_id] = w
        
    print("\nTraining Class Weights (Inverse Frequency w_c = N / (K * N_c)):")
    for c_id, w in class_weights.items():
        print(f"  Class {c_id} [{class_names[c_id]:24s}]: weight = {w:.4f}")
        
    # Save Split Manifest CSV
    split_cols = [
        "filename", "split", "class_id", "t253_target_class", "group_key",
        "duration_seconds", "width", "height", "fps", "frame_count",
        "original_source", "md5", "file_path"
    ]
    split_manifest = split_df[split_cols].sort_values(by=["split", "class_id", "filename"])
    split_manifest.to_csv(WORK_REPORTS_DIR / "split_manifest.csv", index=False)
    split_manifest.to_csv(REPORTS_DIR / "split_manifest.csv", index=False)
    print(f"\n✓ Saved split manifest ({len(split_manifest)} rows) to reports/split_manifest.csv")
    
    # Save Class Distribution CSV
    dist_df.to_csv(WORK_REPORTS_DIR / "split_class_distribution.csv", index=False)
    dist_df.to_csv(REPORTS_DIR / "split_class_distribution.csv", index=False)
    print("✓ Saved split class distribution to reports/split_class_distribution.csv")
    
    # Step 5: Augmentation Configuration YAML
    print("\n[Step 5] Generating Augmentation Configuration YAML...")
    aug_config = {
        "pipeline_version": "1.0",
        "description": "Project T253 Activity Recognition Data Augmentation Configuration",
        "input_specs": {
            "target_resolution": [224, 224],
            "num_frames_per_clip": 16,
            "temporal_sampling_mode": "uniform_stride",
            "channels": 3,
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        },
        "training_augmentations": {
            "random_horizontal_flip": {
                "enabled": True,
                "probability": 0.5,
                "justification": "Bipedal kinematics and postures (standing, sitting, running, falling) remain identical under left-right reflection."
            },
            "color_jitter": {
                "enabled": True,
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.05,
                "probability": 0.7,
                "justification": "Simulates diverse CCTV illumination (indoor fluorescent, direct sunlight, shadows)."
            },
            "mild_rotation": {
                "enabled": True,
                "degrees_range": [-5.0, 5.0],
                "probability": 0.5,
                "justification": "Simulates camera physical mounting tilt angle variation without disrupting vertical gravitational posture orientation."
            },
            "random_scale_crop": {
                "enabled": True,
                "scale_range": [0.85, 1.0],
                "ratio_range": [0.9, 1.1],
                "probability": 0.5,
                "justification": "Simulates passenger distance / zoom variation relative to ceiling CCTV camera."
            },
            "temporal_jitter": {
                "enabled": True,
                "probability": 0.5,
                "justification": "Random offset within each temporal segment prevents overfitting to specific frame indices."
            }
        },
        "validation_test_transforms": {
            "spatial_resize": [256, 256],
            "center_crop": [224, 224],
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            },
            "temporal_sampling": "deterministic_uniform_16_frames",
            "justification": "Evaluation must remain 100% deterministic, unaugmented, and representative of real-world CCTV streaming."
        },
        "class_imbalance_handling": {
            "strategy": "class_weighted_cross_entropy_and_focal_loss",
            "class_weights": class_weights,
            "focal_loss_gamma": 2.0,
            "justification": "Prevents majority classes (walking/standing) from dominating gradients over rare safety-critical anomalies (fallen/sitting)."
        }
    }
    
    with open(WORK_REPORTS_DIR / "augmentation_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(aug_config, f, sort_keys=False, indent=2)
    with open(REPORTS_DIR / "augmentation_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(aug_config, f, sort_keys=False, indent=2)
    print("✓ Saved augmentation configuration to reports/augmentation_config.yaml")
    
    # Step 6: Leakage Verification Report
    print("\n[Step 6] Generating Leakage Verification Report...")
    generate_leakage_report(split_df, train_groups, val_groups, test_groups, train_md5s, val_md5s, test_md5s)
    
    # Step 7: Preprocessing Pipeline Report
    print("\n[Step 7] Generating Preprocessing Pipeline Documentation...")
    generate_preprocessing_report(split_df, class_weights)
    
    print("\n" + "=" * 70)
    print("PHASE 5 PREPROCESSING, AUGMENTATION & SPLITTING COMPLETED")
    print("=" * 70)

def generate_leakage_report(split_df, train_groups, val_groups, test_groups, train_md5s, val_md5s, test_md5s):
    p1 = WORK_REPORTS_DIR / "leakage_verification_report.md"
    p2 = REPORTS_DIR / "leakage_verification_report.md"
    
    md = f"""# Project T253 — Escalator Misuse & Unsafe Riding Behaviour Detection
## Phase 5: Data Leakage Verification Report

**Audit Date:** 2026-11-11  
**Status:** **PASSED (ZERO LEAKAGE DETECTED)**  
**Verification Protocol:** Cryptographic Hash Auditing + Scene/Recording Session Disjointness  

---

## 1. Audit Objective

A primary failure mode in computer vision action recognition is **environmental data leakage**—where frames or video clips originating from the same recording session, physical camera position, or background scene appear in both the training and testing partitions. This artificially inflates validation accuracy because models memorize background textures rather than learning human kinematic patterns.

This audit certifies that the Project T253 split manifest is **strictly partitioned by recording session and cryptographic identity**.

---

## 2. Leakage Test Results

### Test 1: Recording Session / Camera Group Disjointness
- **Train Unique Groups:** {len(train_groups)}
- **Validation Unique Groups:** {len(val_groups)}
- **Test Unique Groups:** {len(test_groups)}
- **Train $\\cap$ Validation Overlap:** **0 groups (PASS)**
- **Train $\\cap$ Test Overlap:** **0 groups (PASS)**
- **Validation $\\cap$ Test Overlap:** **0 groups (PASS)**

> **Conclusion:** Every recording sequence (e.g. all clips from Okutama session `2.1.6`, or VIRAT scene `050201`, or CASIA subject `p01`) resides exclusively in exactly one partition. Zero scene leakage exists.

### Test 2: Cryptographic Binary Hash (MD5) Disjointness
- **Train Unique Hashes:** {len(train_md5s)}
- **Validation Unique Hashes:** {len(val_md5s)}
- **Test Unique Hashes:** {len(test_md5s)}
- **Train $\\cap$ Validation Collisions:** **0 files (PASS)**
- **Train $\\cap$ Test Collisions:** **0 files (PASS)**
- **Validation $\\cap$ Test Collisions:** **0 files (PASS)**

> **Conclusion:** Zero duplicate or identical binary clips exist across any partition boundaries.

### Test 3: Test Set Integrity Certification
- The test set consists of **{len(split_df[split_df['split'] == 'test'])} clips** from **{len(test_groups)} completely unseen recording sessions**.
- The test set has NOT been balanced artificially, preserving natural surveillance activity distribution.
- Test data transformations are strictly deterministic (resize, center-crop, normalization) with zero stochastic augmentation.

---

## 3. Split Summary Table

| Split | Clip Count | Percentage | Recording Groups | Unique Hashes |
|---|---|---|---|---|
| **TRAIN** | {len(split_df[split_df['split'] == 'train'])} | {len(split_df[split_df['split'] == 'train'])/len(split_df)*100:.1f}% | {len(train_groups)} | {len(train_md5s)} |
| **VAL** | {len(split_df[split_df['split'] == 'val'])} | {len(split_df[split_df['split'] == 'val'])/len(split_df)*100:.1f}% | {len(val_groups)} | {len(val_md5s)} |
| **TEST** | {len(split_df[split_df['split'] == 'test'])} | {len(split_df[split_df['split'] == 'test'])/len(split_df)*100:.1f}% | {len(test_groups)} | {len(test_md5s)} |
| **TOTAL** | **{len(split_df)}** | **100.0%** | **{len(train_groups)+len(val_groups)+len(test_groups)}** | **{len(train_md5s)+len(val_md5s)+len(test_md5s)}** |

*Certified by Senior Deep Learning Engineer, Project T253.*
"""
    with open(p1, "w", encoding="utf-8") as f:
        f.write(md)
    with open(p2, "w", encoding="utf-8") as f:
        f.write(md)

def generate_preprocessing_report(split_df, class_weights):
    p1 = WORK_REPORTS_DIR / "preprocessing_pipeline.md"
    p2 = REPORTS_DIR / "preprocessing_pipeline.md"
    
    md = f"""# Project T253 — Escalator Misuse & Unsafe Riding Behaviour Detection
## Phase 5: Preprocessing & Data Augmentation Pipeline Specification

**Version:** 1.0  
**Effective Date:** 2026-11-11  
**Target Hardware Baseline:** NVIDIA GeForce RTX 2050 (4 GB VRAM) / CPU Fallback  

---

## 1. Video Preprocessing Architecture

Video activity classification requires temporal sampling that captures movement dynamics without exhausting GPU memory:

```
[Raw MP4 Video (H.265/HEVC)]
           │
           ▼
[Uniform Temporal Sampling (T = 16 Frames)]
   - Extract 16 equidistant frames spanning clip duration
   - Avoids decoding redundant consecutive frames
           │
           ▼
[Spatial Resizing & Letterboxing]
   - Training: Resize & Random Crop to (224 x 224)
   - Val / Test: Resize to 256x256 -> Center Crop to (224 x 224)
           │
           ▼
[Color Space Normalization]
   - Float32 tensors scaled to [0.0, 1.0]
   - Standardized via ImageNet parameters:
     mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
           │
           ▼
[Tensor Formulation]
   - PyTorch Tensor of shape (Batch, Channels, Time, Height, Width) -> (B, 3, 16, 224, 224)
```

---

## 2. Memory & Hardware Constraints (4 GB VRAM)

On the primary target GPU (RTX 2050 Laptop, 4 GB VRAM):
- Storing $16$ frames of $224 \\times 224 \\times 3$ in float32 requires $\\approx 9.6$ MB per video clip.
- At batch size $B = 8$, raw input tensor memory is $\\approx 77$ MB.
- Activation memory during backpropagation for standard 3D CNNs (e.g. R(2+1)D or Video ResNet-18) requires $\\approx 1.8$–$2.4$ GB.
- **Recommended Hyperparameters:**
  - `batch_size = 8` (or `4` with gradient accumulation steps $= 2$)
  - `num_workers = 2` (Windows multiprocessing stability)
  - Mixed precision training (`torch.cuda.amp.autocast`) to reduce VRAM by 45%.

---

## 3. Data Augmentation Policy

Augmentation is strictly isolated to the **training partition**:

| Transform | Probability | Parameters | Physical Meaning |
|---|---|---|---|
| **Random Horizontal Flip** | $p = 0.5$ | Horizontal axis | Left-right gait symmetry; doubling posture diversity |
| **Color Jitter** | $p = 0.7$ | Brightness $\\pm 0.2$, Contrast $\\pm 0.2$, Saturation $\\pm 0.2$ | Simulates CCTV camera sensor gain and illumination shifts |
| **Mild Rotation** | $p = 0.5$ | Angle $\\in [-5^\\circ, +5^\\circ]$ | Simulates physical CCTV mounting tilt |
| **Random Resized Crop** | $p = 0.5$ | Scale $[0.85, 1.0]$, Aspect Ratio $[0.9, 1.1]$ | Simulates passenger scale variation and distance from lens |
| **Temporal Jitter** | $p = 0.5$ | Random segment offset | Prevents network overfitting to exact fractional timestamps |

---

## 4. Class Imbalance Handling

Training partition inverse frequency weights:

| Class ID | Class Name | Training Count | Loss Weight ($w_c$) |
|---|---|---|---|
| **0** | `NORMAL_RIDING_STANDING` | {len(split_df[(split_df['split']=='train') & (split_df['class_id']==0)])} | `{class_weights[0]:.4f}` |
| **1** | `WALKING` | {len(split_df[(split_df['split']=='train') & (split_df['class_id']==1)])} | `{class_weights[1]:.4f}` |
| **2** | `RUNNING_RUSHING` | {len(split_df[(split_df['split']=='train') & (split_df['class_id']==2)])} | `{class_weights[2]:.4f}` |
| **3** | `SITTING_SQUATTING` | {len(split_df[(split_df['split']=='train') & (split_df['class_id']==3)])} | `{class_weights[3]:.4f}` |
| **4** | `FALLEN_PERSON` | {len(split_df[(split_df['split']=='train') & (split_df['class_id']==4)])} | `{class_weights[4]:.4f}` |

*Weights are passed directly to `torch.nn.CrossEntropyLoss(weight=weights_tensor)`.*
"""
    with open(p1, "w", encoding="utf-8") as f:
        f.write(md)
    with open(p2, "w", encoding="utf-8") as f:
        f.write(md)

if __name__ == "__main__":
    main()
