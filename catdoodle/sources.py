"""Where the real cats come from: Sell Anything's photo labels and the AFHQ face points (paths given, not stored)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SELL_ANYTHING_ML = Path(__file__).resolve().parents[2] / "sell-anything" / "ml"
COAT_JSONL = SELL_ANYTHING_ML / "runs/coat/coat.jsonl"
LANDMARKS_NPZ = SELL_ANYTHING_ML / "data/processed/afhq_cats_mae_pad1.5.npz"


def face_frame(points: np.ndarray) -> np.ndarray:
    """Nine points (pixels) in the photo labels' face frame: origin between the eyes, u along eye 0 -> eye 1,
    v from the mouth towards the eyes, in eye distances (oilprocess.Face)."""
    p = np.asarray(points, float)
    mid = (p[0] + p[1]) / 2
    d = np.linalg.norm(p[1] - p[0])
    right = (p[1] - p[0]) / d
    up = (mid - p[2]) / np.linalg.norm(mid - p[2])
    rel = p - mid
    return np.c_[rel @ right, rel @ up] / d


def load_cats(coat_jsonl: Path = COAT_JSONL, landmarks_npz: Path = LANDMARKS_NPZ):
    """(records, points_uv, split): each cat's photo-label record, its nine points in the face frame, and
    "train" or "test" from AFHQ's own split."""
    lab = np.load(landmarks_npz)
    index = {str(i): k for k, i in enumerate(lab["ids"])}
    records, points, split = [], [], []
    for line in Path(coat_jsonl).read_text().splitlines():
        r = json.loads(line)
        k = index[r["id"]]
        records.append(r)
        points.append(face_frame(lab["points"][k]))
        split.append(r["id"].split("/")[0])
    return records, np.array(points), np.array(split)
