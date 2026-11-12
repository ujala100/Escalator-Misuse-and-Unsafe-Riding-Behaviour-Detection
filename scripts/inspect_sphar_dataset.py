"""
Phase 2: SPHAR Dataset Inspection & Inventory Generator
Project T253 - Escalator Misuse & Unsafe Riding Behaviour Detection

This script inspects all videos in the SPHAR dataset and generates:
  1. reports/dataset_inventory.csv       — per-video metadata
  2. reports/class_distribution.csv      — per-class summary statistics
  3. A console summary of decode compatibility

Run:
    py scripts/inspect_sphar_dataset.py
"""

import csv
import hashlib
import os
import sys
import time
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SPHAR_ROOT = Path(r"C:\Users\ashis\Downloads\SPHAR-Dataset-1.0\SPHAR-Dataset-1.0\videos")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

INVENTORY_CSV = REPORTS_DIR / "dataset_inventory.csv"
CLASS_DIST_CSV = REPORTS_DIR / "class_distribution.csv"

# Source abbreviation → full source name mapping (from SPHAR README filenames)
SOURCE_ABBREVS = {
    "caviar": "CAVIAR",
    "casia": "CASIA",
    "ucfaerial": "UCF-Aerial",
    "ucfcrime": "UCF-Crime",
    "utinteraction": "UT-Interaction",
    "ut": "UT-Interaction",
    "bit": "BIT-Interaction",
    "bitinteraction": "BIT-Interaction",
    "livevideos": "Live Videos",
    "live": "Live Videos",
    "ucfarg": "UCF-ARG",
    "virat": "VIRAT Ground",
    "meva": "MEVA",
    "okutama": "Okutama",
}


def identify_source(filename: str) -> str:
    """Attempt to identify the original dataset source from the filename."""
    fn_lower = filename.lower()
    for abbrev, full_name in SOURCE_ABBREVS.items():
        if abbrev in fn_lower:
            return full_name
    return "unknown"


def compute_md5(filepath: Path, chunk_size: int = 1 << 20) -> str:
    """Compute MD5 hash of a file for duplicate detection."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def inspect_video(filepath: Path) -> dict:
    """
    Open a video file with OpenCV and extract metadata.
    Returns a dict with all inventory fields.
    """
    record = {
        "file_path": str(filepath),
        "filename": filepath.name,
        "class_name": filepath.parent.name,
        "original_source": identify_source(filepath.name),
        "file_size_bytes": filepath.stat().st_size,
        "file_size_mb": round(filepath.stat().st_size / (1024 * 1024), 3),
        "duration_seconds": 0.0,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "frame_count": 0,
        "codec": "",
        "decode_status": "FAIL",
        "first_frame_ok": False,
        "md5": "",
    }

    try:
        cap = cv2.VideoCapture(str(filepath))
        if not cap.isOpened():
            record["decode_status"] = "CANNOT_OPEN"
            return record

        record["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        record["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        record["fps"] = round(cap.get(cv2.CAP_PROP_FPS), 2)
        record["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Decode codec fourcc
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        if fourcc_int > 0:
            record["codec"] = "".join(
                [chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)]
            )
        else:
            record["codec"] = "unknown"

        # Duration
        if record["fps"] > 0 and record["frame_count"] > 0:
            record["duration_seconds"] = round(
                record["frame_count"] / record["fps"], 3
            )

        # Try reading first frame
        ret, frame = cap.read()
        if ret and frame is not None:
            record["first_frame_ok"] = True
            record["decode_status"] = "OK"
        else:
            record["decode_status"] = "NO_FRAMES"

        cap.release()

    except Exception as e:
        record["decode_status"] = f"ERROR: {e}"

    return record


def main():
    print("=" * 70)
    print("SPHAR Dataset Inspection — Phase 2")
    print("=" * 70)

    if not SPHAR_ROOT.exists():
        print(f"ERROR: SPHAR dataset not found at {SPHAR_ROOT}")
        sys.exit(1)

    # Collect all video files
    class_dirs = sorted([d for d in SPHAR_ROOT.iterdir() if d.is_dir()])
    print(f"\nFound {len(class_dirs)} class directories.")

    all_records = []
    class_stats = {}
    total_files = 0
    decode_ok = 0
    decode_fail = 0

    for class_dir in class_dirs:
        class_name = class_dir.name
        videos = sorted(class_dir.glob("*"))
        videos = [v for v in videos if v.is_file()]
        class_count = len(videos)
        total_files += class_count
        print(f"\n  [{class_name}] — {class_count} files")

        class_durations = []
        class_sizes = []
        class_resolutions = set()
        class_sources = {}
        class_ok = 0

        for i, vpath in enumerate(videos):
            rec = inspect_video(vpath)
            all_records.append(rec)

            if rec["decode_status"] == "OK":
                decode_ok += 1
                class_ok += 1
                class_durations.append(rec["duration_seconds"])
                class_sizes.append(rec["file_size_mb"])
                class_resolutions.add(f"{rec['width']}x{rec['height']}")
            else:
                decode_fail += 1

            src = rec["original_source"]
            class_sources[src] = class_sources.get(src, 0) + 1

            if (i + 1) % 100 == 0:
                print(f"    ... processed {i + 1}/{class_count}")

        class_stats[class_name] = {
            "total_videos": class_count,
            "decode_ok": class_ok,
            "decode_fail": class_count - class_ok,
            "min_duration_s": round(min(class_durations), 2) if class_durations else 0,
            "max_duration_s": round(max(class_durations), 2) if class_durations else 0,
            "mean_duration_s": round(
                sum(class_durations) / len(class_durations), 2
            ) if class_durations else 0,
            "total_size_mb": round(sum(class_sizes), 2),
            "resolutions": "; ".join(sorted(class_resolutions)),
            "sources": "; ".join(
                f"{k}({v})" for k, v in sorted(class_sources.items())
            ),
        }

    # -----------------------------------------------------------------------
    # Write dataset_inventory.csv
    # -----------------------------------------------------------------------
    fieldnames = [
        "file_path", "filename", "class_name", "original_source",
        "file_size_bytes", "file_size_mb", "duration_seconds",
        "width", "height", "fps", "frame_count", "codec",
        "decode_status", "first_frame_ok",
    ]
    with open(INVENTORY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\n✓ Inventory written to: {INVENTORY_CSV}")

    # -----------------------------------------------------------------------
    # Write class_distribution.csv
    # -----------------------------------------------------------------------
    dist_fields = [
        "class_name", "total_videos", "decode_ok", "decode_fail",
        "min_duration_s", "max_duration_s", "mean_duration_s",
        "total_size_mb", "resolutions", "sources",
    ]
    with open(CLASS_DIST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=dist_fields)
        writer.writeheader()
        for cls_name in sorted(class_stats.keys()):
            row = {"class_name": cls_name, **class_stats[cls_name]}
            writer.writerows([row])

    print(f"✓ Class distribution written to: {CLASS_DIST_CSV}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total class directories:  {len(class_dirs)}")
    print(f"  Total video files:        {total_files}")
    print(f"  Decode OK:                {decode_ok}")
    print(f"  Decode FAIL:              {decode_fail}")
    print(f"\nPer-class breakdown:")
    for cls in sorted(class_stats.keys()):
        s = class_stats[cls]
        print(
            f"  {cls:15s}  videos={s['total_videos']:5d}  "
            f"ok={s['decode_ok']:5d}  fail={s['decode_fail']:3d}  "
            f"dur={s['min_duration_s']:.1f}–{s['max_duration_s']:.1f}s  "
            f"size={s['total_size_mb']:.1f}MB"
        )
    print("=" * 70)


if __name__ == "__main__":
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f} seconds.")
