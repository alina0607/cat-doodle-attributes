"""The reader: a small convolutional network with one classifier per attribute.

Every head has an "unspecified" class of its own: whether the player has painted the coat or an eye is something
the network decides from the picture, like everything else. The last feature map keeps its 8 × 8 layout (it is not
averaged away), because which region a patch sits in is a question of where.
"""

from __future__ import annotations

import torch
from torch import nn

from .schema import NO_PATCH, CatAttributes, Coat, Eye, Pattern, Region

UNSPEC = "unspecified"
COAT_CLASSES = [c.value for c in Coat] + [UNSPEC]
PATTERN_CLASSES = [p.value for p in Pattern] + [UNSPEC]
EYE_CLASSES = [e.value for e in (Eye.YELLOW, Eye.GREEN, Eye.BLUE, Eye.ORANGE)] + [UNSPEC]   # no copper paint
PATCH_CLASSES = [NO_PATCH] + [c.value for c in Coat if c is not Coat.MIXED] + [UNSPEC]
HEADS = {"coat": COAT_CLASSES, "pattern": PATTERN_CLASSES, "eye_left": EYE_CLASSES, "eye_right": EYE_CLASSES,
         **{f"patch_{r.value}": PATCH_CLASSES for r in Region}}


def encode(a: CatAttributes) -> dict[str, int]:
    """Class indices of an answer, one per head."""
    val = lambda x: UNSPEC if x is None else (x if isinstance(x, str) and not hasattr(x, "value") else x.value)  # noqa: E731
    out = {"coat": COAT_CLASSES.index(val(a.coat)), "pattern": PATTERN_CLASSES.index(val(a.pattern)),
           "eye_left": EYE_CLASSES.index(val(a.eye_left)), "eye_right": EYE_CLASSES.index(val(a.eye_right))}
    for r in Region:
        out[f"patch_{r.value}"] = PATCH_CLASSES.index(val(a.patches[r]))
    return out


def decode(indices: dict[str, int]) -> CatAttributes:
    get = lambda head: (lambda name: None if name == UNSPEC else name)(HEADS[head][indices[head]])  # noqa: E731
    return CatAttributes(coat=get("coat"), pattern=get("pattern"), eye_left=get("eye_left"),
                         eye_right=get("eye_right"), patches={r: get(f"patch_{r.value}") for r in Region})


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.MaxPool2d(2))


class Reader(nn.Module):
    def __init__(self, width: int = 24, dropout: float = 0.2):
        super().__init__()
        w = width
        stem = nn.Sequential(nn.Conv2d(3, w, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(w),
                             nn.ReLU(inplace=True))                                             # 128 -> 64, cheap
        self.features = nn.Sequential(stem, _block(w, 2 * w), _block(2 * w, 4 * w), _block(4 * w, 6 * w),
                                      nn.Conv2d(6 * w, 32, 1), nn.ReLU(inplace=True))          # (32, 8, 8) for 128 px
        self.trunk = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(32 * 8 * 8, 512), nn.ReLU(inplace=True),
                                   nn.Dropout(dropout))
        self.heads = nn.ModuleDict({k: nn.Linear(512, len(v)) for k, v in HEADS.items()})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: (n, 3, 128, 128) in 0–1."""
        h = self.trunk(self.features(x - 0.5))
        return {k: head(h) for k, head in self.heads.items()}
