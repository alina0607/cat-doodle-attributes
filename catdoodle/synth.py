"""Coloured cat doodles drawn from a description, so every training picture comes with its exact answer.

The description is sampled from real cats (one record of Sell Anything's photo labels, runs/coat/coat.jsonl:
its coat, pattern, eye colours and the regions that carry a patch), then varied so rare combinations exist too
(odd eyes, extra patches), and parts are left unpainted, which is what "unspecified" means. The face is laid out
on a real cat's nine face points. The picture is drawn the way a player would with the game's tools — a pencil
outline, flat fills or oil-brush daubs from a 12-colour palette on cream paper, eyes, a nose — and then moved,
scaled and turned. Doodles of anything else, beside the cat, are distractors and never change the answer.

The answer is read from what was actually drawn, not from the sampled description: a region has a patch when at
least PATCH_SHARE of it is painted in a fur colour other than the coat's, the rule the photo labels use. Stripes
and spots are drawn on a separate layer that does not count towards patches, as the photo labels blur them away.

The face frame is the photo labels' one: origin between the eyes, u from the left eye to the right eye, v up, in
eye distances; left and right as seen in the picture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .schema import NO_PATCH, CatAttributes, Coat, Eye, Pattern, Region

SIDE = 128                       # the reader's input
DRAW = 2 * SIDE                  # drawn at twice the size, then reduced, so edges are smooth like the game's
PAPER = (237, 230, 214)
PALETTE = {"black": (32, 30, 30), "white": (250, 250, 248), "grey": (140, 140, 146), "orange": (232, 138, 52),
           "cream": (238, 214, 162), "brown": (118, 74, 44), "yellow": (240, 208, 46), "green": (78, 168, 80),
           "blue": (62, 118, 220), "pink": (238, 150, 170), "red": (206, 52, 50), "purple": (138, 82, 178)}
FUR = ["white", "black", "grey", "orange", "cream", "brown"]
EYE_PAINT = {Eye.YELLOW: "yellow", Eye.GREEN: "green", Eye.BLUE: "blue", Eye.ORANGE: "orange"}
PATCH_SHARE = 0.2
# The photo labels' typical face points in (u, v) (medians over 5,558 AFHQ cats), used when no real layout is given.
MEDIAN_POINTS = np.array([[-0.5, 0], [0.5, 0], [0, -0.81], [-1.15, 0.48], [-1.25, 1.6], [-0.47, 0.96],
                          [0.47, 0.96], [1.25, 1.6], [1.15, 0.48]])


@dataclass
class Doodle:
    image: np.ndarray            # (SIDE, SIDE, 3) uint8
    answer: CatAttributes        # what was drawn
    points: np.ndarray           # (9, 2) the face points in image pixels


# ---------------------------------------------------------------- the face's regions, as in the photo labels

def region_masks(u: np.ndarray, v: np.ndarray, p: np.ndarray) -> dict[Region, np.ndarray]:
    """The photo labels' seven regions on a (u, v) grid, given the nine points p in (u, v)."""
    muzzle = ((u / 0.42) ** 2 + ((v + 0.72) / 0.40) ** 2) <= 1
    ear_l, ear_r = _triangle(u, v, p[3], p[4], p[5]), _triangle(u, v, p[6], p[7], p[8])
    lift = np.array([0.0, 0.35])
    crown = _polygon(u, v, np.array([[-0.5, 0.3], p[5], p[5] + lift, p[6] + lift, p[6], [0.5, 0.3]])) & ~ear_l & ~ear_r
    bridge = (np.abs(u) <= 0.17) & (v >= -0.32) & (v <= 0.3) & ~muzzle
    cheek = (np.abs(u) >= 0.25) & (np.abs(u) <= 1.15) & (v >= -0.95) & (v <= -0.2) & ~muzzle
    return {Region.CROWN: crown, Region.EAR_LEFT: ear_l, Region.EAR_RIGHT: ear_r, Region.BRIDGE: bridge,
            Region.MUZZLE_CHIN: muzzle, Region.CHEEK_LEFT: cheek & (u < 0), Region.CHEEK_RIGHT: cheek & (u > 0)}


def _polygon(u, v, poly):
    inside = np.zeros(u.shape, bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        (ui, vi), (uj, vj) = poly[i], poly[j]
        cross = ((vi > v) != (vj > v)) & (u < (uj - ui) * (v - vi) / (vj - vi + 1e-12) + ui)
        inside ^= cross
        j = i
    return inside


def _triangle(u, v, a, b, c):
    return _polygon(u, v, np.array([a, b, c]))


def head_outline(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A closed outline in (u, v) through the ears and round the cheeks and chin, a little wobbly."""
    side = rng.uniform(1.15, 1.45)
    jaw, chin = rng.uniform(0.8, 1.05), rng.uniform(-1.25, -1.0)
    pts = [p[3], p[4], p[5], p[6], p[7], p[8], [side, -0.35], [jaw, -0.95], [0, chin], [-jaw, -0.95], [-side, -0.35]]
    pts = np.array(pts, float) + rng.normal(0, 0.035, (len(pts), 2))
    out = []
    for i in range(len(pts)):                      # round off the corners a little (Chaikin, two passes)
        out += [pts[i]]
    poly = np.array(out)
    for _ in range(2):
        nxt = np.roll(poly, -1, axis=0)
        poly = np.concatenate([0.75 * poly + 0.25 * nxt, 0.25 * poly + 0.75 * nxt], axis=1).reshape(-1, 2)
    return poly


# ---------------------------------------------------------------- sampling a description

def sample_description(record: dict | None, rng: np.random.Generator) -> dict:
    """What to draw, as plain names: coat base colour, pattern, eye names, patch colour per region."""
    if record is None:
        base = str(rng.choice(FUR))
        record = {"base_colour": base, "pattern": rng.choice([p.value for p in Pattern]),
                  "eye_left": {"name": "yellow"}, "eye_right": {"name": "yellow"}, "regions": {}}
    base = record["base_colour"] if record["base_colour"] in FUR else "brown"
    patches = {r.value: (reg["patch_name"] if reg.get("has_patch") and reg.get("patch_name") in FUR else None)
               for r in Region if (reg := record["regions"].get(r.value, {})) is not None}
    if rng.random() < 0.2:                         # an extra patch somewhere, in another colour
        patches[str(rng.choice([r.value for r in Region]))] = str(rng.choice([c for c in FUR if c != base]))
    eyes = []
    for k in ("eye_left", "eye_right"):
        name = record[k]["name"] if isinstance(record.get(k), dict) else "unclear"
        eyes.append({"copper": "orange"}.get(name, name))     # the palette has no copper
    if rng.random() < 0.15:                        # odd eyes, rare in photos
        eyes[rng.integers(2)] = str(rng.choice(list(EYE_PAINT.values())))
    if rng.random() < 0.1:
        eyes = [str(rng.choice(list(EYE_PAINT.values())))] * 2
    return {"base": base, "pattern": record["pattern"], "eyes": eyes, "patches": patches}


# ---------------------------------------------------------------- drawing

class _Canvas:
    """Draws in (u, v) on a DRAW × DRAW picture, plus a label layer of palette indices for fur."""

    def __init__(self, rng, centre, scale, angle):
        self.img = Image.new("RGB", (DRAW, DRAW), PAPER)
        self.draw = ImageDraw.Draw(self.img)
        self.fur = Image.new("L", (DRAW, DRAW), 0)          # 0 = not fur, else 1 + FUR index
        self.fur_draw = ImageDraw.Draw(self.fur)
        self.rng, self.c, self.s = rng, np.asarray(centre, float), scale
        self.cos, self.sin = math.cos(angle), math.sin(angle)

    def px(self, uv) -> list[tuple[float, float]]:
        uv = np.atleast_2d(np.asarray(uv, float))
        x = self.c[0] + self.s * (uv[:, 0] * self.cos - uv[:, 1] * self.sin)
        y = self.c[1] - self.s * (uv[:, 0] * self.sin + uv[:, 1] * self.cos)
        return list(zip(x.tolist(), y.tolist(), strict=True))

    def fill(self, poly_uv, colour, brushy, fur_index=0):
        pts = self.px(poly_uv)
        if brushy:                                  # oil-brush daubs: thick strokes inside, some paper showing
            mask = Image.new("L", (DRAW, DRAW), 0)
            ImageDraw.Draw(mask).polygon(pts, fill=255)
            layer = Image.new("RGB", (DRAW, DRAW), PALETTE[colour])
            daubs = Image.new("L", (DRAW, DRAW), 0)
            dd = ImageDraw.Draw(daubs)
            xs, ys = zip(*pts, strict=True)
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            w = int(self.rng.integers(8, 18))
            for _ in range(int(area / (w * w) * 1.6) + 4):
                x, y = self.rng.uniform(min(xs), max(xs)), self.rng.uniform(min(ys), max(ys))
                a, ln = self.rng.uniform(0, math.pi), self.rng.uniform(1.5, 4) * w
                dd.line([(x - ln * math.cos(a) / 2, y - ln * math.sin(a) / 2),
                         (x + ln * math.cos(a) / 2, y + ln * math.sin(a) / 2)], fill=255, width=w)
            covered = Image.fromarray((np.asarray(daubs) & np.asarray(mask)).astype(np.uint8))
            self.img.paste(layer, mask=covered)
            if fur_index:
                self.fur.paste(fur_index, mask=covered)
        else:
            self.draw.polygon(pts, fill=PALETTE[colour])
            if fur_index:
                self.fur_draw.polygon(pts, fill=fur_index)

    def line(self, uv, colour, width):
        self.draw.line(self.px(uv), fill=PALETTE[colour], width=width, joint="curve")

    def ellipse(self, centre_uv, ru, rv, colour, outline=None, width=2):
        pts = [(centre_uv[0] + ru * math.cos(t), centre_uv[1] + rv * math.sin(t))
               for t in np.linspace(0, 2 * math.pi, 24)]
        self.draw.polygon(self.px(pts), fill=None if colour is None else PALETTE[colour],
                          outline=None if outline is None else PALETTE[outline], width=width)


def _blob(centre, r, rng, heart=False) -> np.ndarray:
    t = np.linspace(0, 2 * math.pi, 28, endpoint=False)
    if heart:                                       # the classic heart curve, scaled to radius r
        x, y = 16 * np.sin(t) ** 3, 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        return np.c_[centre[0] + r * x / 17, centre[1] + r * y / 17]
    rr = r * (1 + 0.25 * np.sin(3 * t + rng.uniform(0, 6)) + rng.normal(0, 0.06, len(t)))
    return np.c_[centre[0] + rr * np.cos(t), centre[1] + 0.8 * rr * np.sin(t)]


REGION_SPOT = {Region.CROWN: ((0, 0.75), 0.42), Region.EAR_LEFT: (None, 0.3), Region.EAR_RIGHT: (None, 0.3),
               Region.BRIDGE: ((0, -0.05), 0.2), Region.MUZZLE_CHIN: ((0, -0.75), 0.42),
               Region.CHEEK_LEFT: ((-0.68, -0.55), 0.42), Region.CHEEK_RIGHT: ((0.68, -0.55), 0.42)}


def draw_doodle(rng: np.random.Generator, record: dict | None = None, points_uv: np.ndarray | None = None,
                distractors: bool = True) -> Doodle:
    """One doodle and its answer. record: a photo-label record to start from (None: anything goes)."""
    desc = sample_description(record, rng)
    p = (MEDIAN_POINTS if points_uv is None else np.asarray(points_uv, float)) + rng.normal(0, 0.04, (9, 2))
    scale = rng.uniform(0.18, 0.32) * DRAW          # pixels per eye distance: the head is 45–85 % of the picture
    centre = (DRAW / 2 + rng.uniform(-0.12, 0.12) * DRAW, DRAW / 2 + rng.uniform(-0.05, 0.15) * DRAW)
    cv = _Canvas(rng, centre, scale, rng.uniform(-0.26, 0.26))
    outline = head_outline(p, rng)
    painted = rng.random() > 0.2                    # 1 in 5: only an outline, the coat left unspecified
    brushy = rng.random() < 0.4
    base = desc["base"]
    if painted:
        cv.fill(outline, base, brushy, 1 + FUR.index(base))
        if desc["pattern"] in ("bicolour", "tortie_calico"):       # mixed coats: big blocks of other colours
            others = ["white"] if desc["pattern"] == "bicolour" else ["orange", "black", "cream"]
            if desc["pattern"] == "bicolour" and base == "white":
                others = [rng.choice(["black", "grey", "orange", "brown"])]
            for _ in range(int(rng.integers(2, 5))):
                r = list(Region)[rng.integers(len(Region))]
                col = str(rng.choice([o for o in others if o != base] or ["white"]))
                desc["patches"][r.value] = col
        for r in Region:
            col = desc["patches"].get(r.value)
            if col and col != base:
                spot, rad = REGION_SPOT[r]
                if spot is None:                    # an ear: fill most of its triangle
                    tri = p[[3, 4, 5]] if r is Region.EAR_LEFT else p[[6, 7, 8]]
                    mid = tri.mean(0)
                    poly = mid + (tri - mid) * rng.uniform(0.6, 0.95)
                else:
                    poly = _blob(np.array(spot) + rng.normal(0, 0.06, 2), rad * rng.uniform(0.8, 1.3), rng,
                                 heart=r is Region.CROWN and rng.random() < 0.25)
                cv.fill(poly, col, brushy and rng.random() < 0.7, 1 + FUR.index(col))
        dark = "black" if base not in ("black", "brown") else ("brown" if base == "black" else "black")
        if desc["pattern"] == "tabby":              # the forehead M, lines from the eyes, cheek bars
            w = int(rng.integers(3, 7))
            for k in (-1, 0, 1):
                cv.line([(0.22 * k, 0.4), (0.3 * k + rng.normal(0, 0.05), 1.0)], dark, w)
            for sgn in (-1, 1):
                cv.line([(sgn * 0.75, 0.0), (sgn * 1.15, 0.12)], dark, w)
                for dv in (-0.35, -0.6):
                    cv.line([(sgn * 0.45, dv), (sgn * 1.2, dv + rng.normal(0, 0.08))], dark, w)
        elif desc["pattern"] == "spots":
            for _ in range(int(rng.integers(4, 9))):
                c = (rng.uniform(-1, 1), rng.uniform(-0.8, 1.0))
                cv.ellipse(c, 0.08, 0.07, dark)
    # eyes: coloured, or just outlined (unspecified)
    eyes_drawn = []
    for i, name in enumerate(desc["eyes"]):
        c = p[i]
        ru, rv = 0.2 * rng.uniform(0.85, 1.2), 0.15 * rng.uniform(0.85, 1.25)
        colour = name if name in EYE_PAINT.values() and rng.random() > 0.25 else None
        cv.ellipse(c, ru, rv, colour if colour else ("white" if rng.random() < 0.5 or not painted else None),
                   outline="black", width=3)
        if colour:
            cv.ellipse(c, 0.035, 0.12, "black")
        eyes_drawn.append(colour)
    cv.line([(p[2][0] - 0.12, -0.5), (p[2][0] + 0.12, -0.5), (p[2][0], -0.62), (p[2][0] - 0.12, -0.5)],
            "pink" if rng.random() < 0.7 else "black", 4)
    cv.line([(-0.18, -0.85), (0, -0.72), (0.18, -0.85)], "black", 3)
    cv.line(np.vstack([outline, outline[:1]]), "black", int(rng.integers(3, 7)))
    for sgn in (-1, 1):
        if rng.random() < 0.6:
            for dv in (-0.62, -0.72):
                cv.line([(sgn * 0.35, dv), (sgn * 1.45, dv + rng.normal(0, 0.12))], "black", 2)
    if distractors and rng.random() < 0.5:
        for _ in range(int(rng.integers(1, 4))):
            x, y = rng.choice([rng.uniform(0, 0.15), rng.uniform(0.85, 1.0)]) * DRAW, rng.uniform(0, 1) * DRAW
            col = str(rng.choice(list(PALETTE)))
            if rng.random() < 0.5:
                cv.draw.ellipse([x - 14, y - 14, x + 14, y + 14], fill=PALETTE[col])
            else:
                cv.draw.line([(x, y), (x + rng.normal(0, 30), y + rng.normal(0, 30))], fill=PALETTE[col], width=5)
    answer = _read_back(cv, p, outline, painted, base, desc, eyes_drawn)
    img = cv.img.filter(ImageFilter.GaussianBlur(0.6)).resize((SIDE, SIDE), Image.LANCZOS)
    pts = np.array(cv.px(p)) * SIDE / DRAW
    return Doodle(np.asarray(img, np.uint8), answer, pts)


def _read_back(cv, p, outline, painted, base, desc, eyes) -> CatAttributes:
    """The answer from what is on the canvas: the fur layer inside each region, the eyes as painted."""
    if not painted:
        return CatAttributes(eye_left=eyes[0], eye_right=eyes[1])
    fur = np.asarray(cv.fur)
    ys, xs = np.mgrid[0:DRAW, 0:DRAW] + 0.5
    dx, dy = (xs - cv.c[0]) / cv.s, -(ys - cv.c[1]) / cv.s
    u, v = dx * cv.cos + dy * cv.sin, -dx * cv.sin + dy * cv.cos
    for c in p[:2]:                                 # the eyes are not fur
        fur = np.where((u - c[0]) ** 2 + (v - c[1]) ** 2 < 0.34 ** 2, 0, fur)
    base_i = 1 + FUR.index(base)
    patches = {}
    for r, m in region_masks(u, v, p).items():
        vals = fur[m & (fur > 0)]
        if len(vals) == 0:
            patches[r] = NO_PATCH
            continue
        other = vals[vals != base_i]
        if len(other) >= PATCH_SHARE * m.sum() and len(other) > 0:
            patches[r] = Coat(FUR[np.bincount(other).argmax() - 1])
        else:
            patches[r] = NO_PATCH
    pattern = Pattern(desc["pattern"]) if desc["pattern"] in [x.value for x in Pattern] else Pattern.SOLID
    coat = Coat.MIXED if pattern in (Pattern.BICOLOUR, Pattern.TORTIE_CALICO) else Coat(base)
    return CatAttributes(coat=coat, pattern=pattern, eye_left=eyes[0], eye_right=eyes[1], patches=patches)
