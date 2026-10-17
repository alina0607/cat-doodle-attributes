"""What the reader reports about a cat, and how a left-right mirror changes it.

Left and right are as seen in the image, the convention of cat-face-landmarks (catlandmarks.landmarks): the
left eye is the one on the left of the picture. Every left/right pair here — eyes, cheeks, ears — swaps under a
mirror, the same way the landmark permutation FLIP_PERMUTATION swaps the nine points, so a mirrored doodle and
its mirrored description always agree.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from enum import Enum

import numpy as np

SHAPE_GRID = 16                    # a patch's shape: a SHAPE_GRID × SHAPE_GRID mask in face coordinates


class Coat(str, Enum):
    WHITE = "white"
    BLACK = "black"
    ORANGE = "orange"
    GREY = "grey"
    CREAM = "cream"
    MIXED = "mixed"


class Pattern(str, Enum):
    SOLID = "solid"
    TABBY = "tabby"                # stripes
    PATCHED = "patched"            # spots or patches
    BICOLOUR = "bicolour"
    TORTOISESHELL = "tortoiseshell"  # tortoiseshell or calico


class Region(str, Enum):
    CROWN = "crown"                # between the ears, above the eyes
    LEFT_CHEEK = "left_cheek"
    RIGHT_CHEEK = "right_cheek"
    NOSE_BRIDGE = "nose_bridge"
    MUZZLE_CHIN = "muzzle_chin"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"


MIRROR_REGION = {Region.LEFT_CHEEK: Region.RIGHT_CHEEK, Region.RIGHT_CHEEK: Region.LEFT_CHEEK,
                 Region.LEFT_EAR: Region.RIGHT_EAR, Region.RIGHT_EAR: Region.LEFT_EAR}

Lab = tuple[float, float, float]   # CIE L*a*b*: L in 0–100, a and b roughly −128–127


def _lab(value) -> Lab:
    lab = tuple(float(v) for v in value)
    if len(lab) != 3 or not 0 <= lab[0] <= 100:
        raise ValueError(f"not a L*a*b* colour: {value}")
    return lab


@dataclass(frozen=True)
class Patch:
    """A patch of a colour other than the coat's, inside one region."""

    colour: Lab
    coverage: float                # share of the region it covers, 0–1
    shape: np.ndarray | None = field(default=None, compare=False)   # (SHAPE_GRID, SHAPE_GRID) bool, face coords

    def __post_init__(self):
        object.__setattr__(self, "colour", _lab(self.colour))
        if not 0 <= self.coverage <= 1:
            raise ValueError(f"coverage {self.coverage} is not in 0–1")
        if self.shape is not None:
            shape = np.asarray(self.shape, bool)
            if shape.shape != (SHAPE_GRID, SHAPE_GRID):
                raise ValueError(f"shape must be {SHAPE_GRID} × {SHAPE_GRID}, got {shape.shape}")
            object.__setattr__(self, "shape", shape)

    def mirrored(self) -> Patch:
        """The shape flips left to right; face coordinates put x across the face, so its columns reverse."""
        return replace(self, shape=None if self.shape is None else self.shape[:, ::-1].copy())


@dataclass(frozen=True)
class CatAttributes:
    coat: Coat
    coat_colour: Lab
    pattern: Pattern
    left_eye: Lab
    right_eye: Lab
    patches: dict[Region, Patch] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "coat", Coat(self.coat))
        object.__setattr__(self, "pattern", Pattern(self.pattern))
        for name in ("coat_colour", "left_eye", "right_eye"):
            object.__setattr__(self, name, _lab(getattr(self, name)))
        object.__setattr__(self, "patches", {Region(k): v for k, v in self.patches.items()})

    def mirrored(self) -> CatAttributes:
        """The description of the mirror image: left and right eyes, cheeks and ears swap."""
        return replace(self, left_eye=self.right_eye, right_eye=self.left_eye,
                       patches={MIRROR_REGION.get(r, r): p.mirrored() for r, p in self.patches.items()})

    def to_json(self) -> str:
        d = asdict(self)
        d["coat"], d["pattern"] = self.coat.value, self.pattern.value
        d["patches"] = {r.value: {"colour": list(p.colour), "coverage": p.coverage,
                                  "shape": None if p.shape is None else p.shape.astype(int).tolist()}
                        for r, p in self.patches.items()}
        return json.dumps(d)

    @classmethod
    def from_json(cls, text: str) -> CatAttributes:
        d = json.loads(text)
        d["patches"] = {Region(r): Patch(colour=p["colour"], coverage=p["coverage"],
                                         shape=None if p["shape"] is None else np.array(p["shape"], bool))
                        for r, p in d["patches"].items()}
        return cls(**d)
