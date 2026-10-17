"""What the reader reports about a cat, and how a left-right mirror changes it.

The names are the ones Sell Anything's photo labels use (ml/tools/coat_attributes.py), so a description measured on
a photo, read from a doodle and given to the painting model is the same object. Every field may be None:
**unspecified** — the player has not drawn that part yet — which is different from "no patch" or "white".

Left and right are as seen in the image, the convention of cat-face-landmarks (catlandmarks.landmarks): the
left eye is the one on the left of the picture. Every left/right pair here — eyes, cheeks, ears — swaps under a
mirror, the same way the landmark permutation FLIP_PERMUTATION swaps the nine points, so a mirrored doodle and
its mirrored description always agree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import Enum


class Coat(str, Enum):
    WHITE = "white"
    BLACK = "black"
    GREY = "grey"
    ORANGE = "orange"
    CREAM = "cream"
    BROWN = "brown"
    MIXED = "mixed"                # bicolour or tortoiseshell/calico: no single coat colour


FUR_COLOURS = [c for c in Coat if c is not Coat.MIXED]      # what a patch can be


class Pattern(str, Enum):
    SOLID = "solid"
    TABBY = "tabby"                # stripes
    SPOTS = "spots"                # spots or blotches
    BICOLOUR = "bicolour"
    TORTIE_CALICO = "tortie_calico"


class Eye(str, Enum):
    YELLOW = "yellow"
    GREEN = "green"
    BLUE = "blue"
    ORANGE = "orange"
    COPPER = "copper"


class Region(str, Enum):
    CROWN = "crown"                # between the ears, above the eyes
    EAR_LEFT = "ear_left"
    EAR_RIGHT = "ear_right"
    BRIDGE = "bridge"              # the nose bridge, between the eyes and the muzzle
    MUZZLE_CHIN = "muzzle_chin"
    CHEEK_LEFT = "cheek_left"
    CHEEK_RIGHT = "cheek_right"


MIRROR_REGION = {Region.CHEEK_LEFT: Region.CHEEK_RIGHT, Region.CHEEK_RIGHT: Region.CHEEK_LEFT,
                 Region.EAR_LEFT: Region.EAR_RIGHT, Region.EAR_RIGHT: Region.EAR_LEFT}
NO_PATCH = "none"


@dataclass(frozen=True)
class CatAttributes:
    """None anywhere means unspecified. A patch is a fur colour, NO_PATCH, or None."""

    coat: Coat | None = None
    pattern: Pattern | None = None
    eye_left: Eye | None = None
    eye_right: Eye | None = None
    patches: dict[Region, Coat | str | None] = field(default_factory=lambda: {r: None for r in Region})

    def __post_init__(self):
        object.__setattr__(self, "coat", None if self.coat is None else Coat(self.coat))
        object.__setattr__(self, "pattern", None if self.pattern is None else Pattern(self.pattern))
        for name in ("eye_left", "eye_right"):
            v = getattr(self, name)
            object.__setattr__(self, name, None if v is None else Eye(v))
        patches = {r: None for r in Region}
        for k, v in self.patches.items():
            if v not in (None, NO_PATCH) and Coat(v) is Coat.MIXED:
                raise ValueError("a patch is one fur colour, not mixed")
            patches[Region(k)] = v if v in (None, NO_PATCH) else Coat(v)
        object.__setattr__(self, "patches", patches)

    def mirrored(self) -> CatAttributes:
        """The description of the mirror image: left and right eyes, cheeks and ears swap."""
        return replace(self, eye_left=self.eye_right, eye_right=self.eye_left,
                       patches={MIRROR_REGION.get(r, r): p for r, p in self.patches.items()})

    def to_dict(self) -> dict:
        val = lambda x: None if x is None else (x if isinstance(x, str) and not isinstance(x, Enum) else x.value)  # noqa: E731
        return {"coat": val(self.coat), "pattern": val(self.pattern), "left_eye": val(self.eye_left),
                "right_eye": val(self.eye_right), "patches": {r.value: val(p) for r, p in self.patches.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> CatAttributes:
        return cls(coat=d.get("coat"), pattern=d.get("pattern"), eye_left=d.get("left_eye"),
                   eye_right=d.get("right_eye"), patches=d.get("patches", {}))

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> CatAttributes:
        return cls.from_dict(json.loads(text))
