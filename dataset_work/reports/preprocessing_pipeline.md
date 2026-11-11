# Project T253 — Escalator Misuse & Unsafe Riding Behaviour Detection
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
- Storing $16$ frames of $224 \times 224 \times 3$ in float32 requires $\approx 9.6$ MB per video clip.
- At batch size $B = 8$, raw input tensor memory is $\approx 77$ MB.
- Activation memory during backpropagation for standard 3D CNNs (e.g. R(2+1)D or Video ResNet-18) requires $\approx 1.8$–$2.4$ GB.
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
| **Color Jitter** | $p = 0.7$ | Brightness $\pm 0.2$, Contrast $\pm 0.2$, Saturation $\pm 0.2$ | Simulates CCTV camera sensor gain and illumination shifts |
| **Mild Rotation** | $p = 0.5$ | Angle $\in [-5^\circ, +5^\circ]$ | Simulates physical CCTV mounting tilt |
| **Random Resized Crop** | $p = 0.5$ | Scale $[0.85, 1.0]$, Aspect Ratio $[0.9, 1.1]$ | Simulates passenger scale variation and distance from lens |
| **Temporal Jitter** | $p = 0.5$ | Random segment offset | Prevents network overfitting to exact fractional timestamps |

---

## 4. Class Imbalance Handling

Training partition inverse frequency weights:

| Class ID | Class Name | Training Count | Loss Weight ($w_c$) |
|---|---|---|---|
| **0** | `NORMAL_RIDING_STANDING` | 1093 | `0.6670` |
| **1** | `WALKING` | 1817 | `0.4012` |
| **2** | `RUNNING_RUSHING` | 327 | `2.2294` |
| **3** | `SITTING_SQUATTING` | 320 | `2.2781` |
| **4** | `FALLEN_PERSON` | 88 | `8.2841` |

*Weights are passed directly to `torch.nn.CrossEntropyLoss(weight=weights_tensor)`.*
