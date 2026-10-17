"""Score the reader on the exam: real doodles drawn in the game and saved with the answer their author confirmed.

The game's "save this doodle" button writes one folder per doodle under data/exam/ (or CATDOODLE_EXAM_DIR):

    board.png         the doodle board as drawn (cream paper)
    answer.json       the attributes the author confirmed, CatAttributes.to_dict() form, null = unspecified
    prediction.json   what the reader said when the doodle was saved, and its version
    edited.json       which fields the author changed from that prediction
    strokes.json, meta.json

The form starts from the reader's prediction, so a field the author did not change is one they confirmed. Both
kinds are counted and reported apart: if the confirmed fields score far higher than the changed ones that is
expected, but the share of changed fields is itself the error rate the author saw.

The exam is never training data: catdoodle.train draws its doodles, it reads no folder, and this module is the only
reader of data/exam/.

    python -m catdoodle.exam                    # the predictions saved with each doodle
    python -m catdoodle.exam --rerun            # the current weights, on the same doodles
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .schema import Region

EXAM_DIR = Path(os.environ.get("CATDOODLE_EXAM_DIR", Path(__file__).resolve().parents[1] / "data" / "exam"))
FIELDS = ["coat", "pattern", "left_eye", "right_eye"] + [f"patches.{r.value}" for r in Region]


def _get(d: dict, field: str):
    if field.startswith("patches."):
        return d.get("patches", {}).get(field.split(".", 1)[1])
    return d.get(field)


def load_exam(folder: Path = EXAM_DIR) -> list[dict]:
    items = []
    for sub in sorted(p for p in Path(folder).glob("*") if (p / "answer.json").exists()):
        item = {"id": sub.name, "answer": json.loads((sub / "answer.json").read_text()),
                "prediction": json.loads((sub / "prediction.json").read_text())
                if (sub / "prediction.json").exists() else {},
                "edited": json.loads((sub / "edited.json").read_text()) if (sub / "edited.json").exists() else {},
                "board": sub / "board.png"}
        items.append(item)
    return items


def score(items: list[dict], rerun: bool = False) -> dict:
    hits = defaultdict(lambda: {"right": 0, "total": 0, "changed": 0, "right_confirmed": 0, "confirmed": 0})
    for it in items:
        if rerun:
            from .reader import read_attributes
            pred = read_attributes(np.asarray(Image.open(it["board"]).convert("RGB")))
        else:
            pred = it["prediction"].get("attributes", it["prediction"])
        for f in FIELDS:
            truth, guess = _get(it["answer"], f), _get(pred, f)
            changed = bool(_get(it["edited"], f)) if it["edited"] else None
            h = hits[f]
            h["total"] += 1
            h["right"] += truth == guess
            if changed is not None:
                h["changed"] += changed
                if not changed:
                    h["confirmed"] += 1
                    h["right_confirmed"] += truth == guess
    return {f: {**h, "accuracy": h["right"] / h["total"] if h["total"] else None} for f, h in hits.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true", help="read every board again with the current weights")
    ap.add_argument("--dir", type=Path, default=EXAM_DIR)
    a = ap.parse_args()
    items = load_exam(a.dir)
    print(f"{len(items)} doodles in {a.dir}")
    if not items:
        return
    for f, s in score(items, a.rerun).items():
        print(f"{f:22s} {s['right']:3d}/{s['total']:3d} = {s['accuracy']:.2f}   changed by the author: {s['changed']}")


if __name__ == "__main__":
    main()
