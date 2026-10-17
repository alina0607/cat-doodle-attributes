"""A sheet of synthetic doodles with their answers, to look at.  python scripts/sample_sheet.py out.png"""

import sys

import numpy as np
from PIL import Image, ImageDraw

from catdoodle.sources import load_cats
from catdoodle.synth import SIDE, draw_doodle

ZH = {None: "?", "none": "-"}


def caption(a):
    d = a.to_dict()
    pat = [f"{k[:5]}:{v[:3]}" for k, v in d["patches"].items() if v not in (None, "none")]
    return [f"{d['coat'] or '?'} {d['pattern'] or '?'}", f"L {d['left_eye'] or '?'} R {d['right_eye'] or '?'}",
            " ".join(pat[:3])]


records, points, split = load_cats()
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
cols, rows = 8, 4
sheet = Image.new("RGB", (cols * SIDE, rows * (SIDE + 40)), "white")
dr = ImageDraw.Draw(sheet)
for k in range(cols * rows):
    i = rng.integers(len(records))
    d = draw_doodle(rng, records[i], points[i])
    x, y = (k % cols) * SIDE, (k // cols) * (SIDE + 40)
    sheet.paste(Image.fromarray(d.image), (x, y))
    for j, t in enumerate(caption(d.answer)):
        dr.text((x + 2, y + SIDE + 1 + 12 * j), t, fill=(0, 0, 0))
sheet.save(sys.argv[1])
print("saved", sys.argv[1])
