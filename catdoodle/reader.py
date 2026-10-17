"""Read a doodle: the entry point the game server calls.

    from catdoodle.reader import read_attributes
    read_attributes(rgb)    # (H, W, 3) uint8, the doodle board on its cream paper -> dict, None = unspecified

The weights are looked up in CATDOODLE_WEIGHTS, else runs/v0/reader.pt beside this package. The picture is
resized to the reader's 128 × 128 the way the synthetic doodles were (Lanczos).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .model import HEADS, Reader, decode
from .synth import SIDE

VERSION = "v0"
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "runs" / VERSION / "reader.pt"


@lru_cache(maxsize=1)
def _model() -> Reader:
    path = Path(os.environ.get("CATDOODLE_WEIGHTS", DEFAULT_WEIGHTS))
    if not path.exists():
        raise FileNotFoundError(f"no reader weights at {path}")
    model = Reader()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


def read_attributes(rgb: np.ndarray) -> dict:
    """The attributes of one doodle, as CatAttributes.to_dict(): None where the reader says unspecified."""
    img = Image.fromarray(np.asarray(rgb, np.uint8)[..., :3]).resize((SIDE, SIDE), Image.LANCZOS)
    x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float()[None] / 255
    with torch.no_grad():
        out = _model()(x)
    return decode({h: int(out[h][0].argmax()) for h in HEADS}).to_dict()
