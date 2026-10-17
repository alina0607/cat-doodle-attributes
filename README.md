# Cat Doodle Attributes

> A model that reads a coloured cat doodle and reports what the cat looks like: its coat colour and pattern,
> which parts of the face carry a patch and in what colour, and the colour of each eye. Its output is the
> condition for a generative model that paints the same cat in oil, with its own light and shade.

**Status: reader v0 trained on synthetic doodles.** It reads doodles drawn by this repository's own generator
well (table below). How well it reads doodles drawn by people is not measured yet: that is the job of the exam
set (see [The exam](#the-exam-real-doodles)), which is still being collected.

## Where it fits

This model serves [Sell Anything](https://github.com/alina0607/sell-anything), a game whose cats are painted
stroke by stroke by an autoregressive transformer (Brush GPT). In a level under construction the player doodles a
cat, and the game paints an oil cat that follows the doodle's *description*, not its pixels:

```
player's doodle
  └─► is it a cat?                       sketch-recognition-from-scratch  (sketch classifier, pencil outline only)
        └─► what does the cat look like?  cat-doodle-attributes            (this repository, the whole coloured board)
              └─► paint that cat           sell-anything                    (Brush GPT, conditioned on the attributes; planned)
```

A white cat with a brown patch on the crown, a yellow left eye and a blue right eye should come out as an oil
painting of a white cat with a brown crown patch and those two eye colours, wherever the player drew it and
however large. The painting's shading and fur direction are the painting model's own: the doodle only says
*what*, never *how*.

## What it reads

| Field | Values |
|---|---|
| `coat` | white, black, grey, orange, cream, brown, mixed (bicolour and tortoiseshell / calico) |
| `pattern` | solid, tabby, spots, bicolour, tortie_calico |
| `left_eye`, `right_eye` | yellow, green, blue, orange (the photo labels also have copper; the game's palette has no copper paint) |
| `patches` for 7 regions | `none` or a fur colour; regions: crown, ear_left, ear_right, bridge, muzzle_chin, cheek_left, cheek_right |

Every field can also be **unspecified** (`None`): the player has not painted that part yet. Unspecified is not
the same as "no patch" or "white".

**Left and right are as seen in the image**, the convention of
[cat-face-landmarks](https://github.com/alina0607/cat-face-landmarks): `left_eye` is the eye on the left of the
picture. Regions are placed in a face frame (origin between the eyes, one unit = the eye distance), so the same
description fits a photo, a painting and a doodle of any size and position. A mirror image swaps the left and
right eyes, ears and cheeks (`CatAttributes.mirrored`), with the permutation the landmark model uses.

Colours drawn **outside** the cat's face (a background, a ball next to the cat, scribbles) are not the cat and
must not change the reading.

## How it is built

1. **A description per real cat.** `catdoodle.sources` loads, for each of AFHQ's 5,558 cats, the attributes
   Sell Anything measured on the photo (`ml/tools/coat_attributes.py`) and the cat's nine face points.
   Photo labels agree with a person's labels only about two times in three, so they are used as a *distribution
   of descriptions*, not as answers.
2. **Synthetic doodles with exact answers** (`catdoodle.synth`). A cat's description and face layout are drawn
   as a doodle on cream paper with the game's paints: outline, flat or brush-stroke fill, patches (sometimes
   heart-shaped), tabby marks, eyes with slit pupils, and with probability 0.5 one to three distractors outside
   the face. Rare combinations (odd eyes, extra patches) are added at fixed rates. The answer is **read back from
   what was drawn**, not taken from the description: a region has a patch when at least 20% of it carries a fur
   colour other than the coat's. An eye left unpainted, or a coat left as an outline, is unspecified.
3. **The reader** (`catdoodle.model`). A small CNN, about 1.56 M parameters: a stride-2 5×5 stem, three blocks of
   two 3×3 convolutions with batch norm, ReLU and max pooling, a 1×1 convolution to 32 × 8 × 8, then a 512-unit
   fully connected layer and 11 linear heads. The 8 × 8 map is flattened rather than averaged, because which
   region a patch is in depends on where it is. Each head has its own "unspecified" class, so whether a part is
   painted is also read by the model.
4. **Training** (`catdoodle.train`). Four doodles per AFHQ *train* cat (20,260); validation doodles come from
   AFHQ *test* cats (1,972), so no validation doodle shares a cat's description or face layout with training.
   The loss is the mean cross-entropy over the 11 heads; half of each batch is mirrored, answers included.
   AdamW, one-cycle learning rate (peak 2e-3), batch 128, 12 epochs on the CPU.

Nothing in the reading is a hand-written rule standing in for the model: the attributes come from the model,
trained on data, as the Sell Anything project requires.

## Results: reader v0 on synthetic validation doodles

From `runs/v0/report.json` (1,972 doodles of held-out cats). *Baseline* always answers the head's most
common training class.

| Head | Accuracy | Baseline |
|---|---|---|
| coat | 0.965 | 0.208 |
| pattern | 0.948 | 0.339 |
| left_eye / right_eye | 0.998 / 0.995 | 0.439 / 0.408 |
| patch: crown | 0.991 | 0.596 |
| patch: ear_left / ear_right | 0.993 / 0.992 | 0.634 / 0.618 |
| patch: bridge | 0.971 | 0.570 |
| patch: muzzle_chin | 0.965 | 0.535 |
| patch: cheek_left / cheek_right | 0.991 / 0.992 | 0.590 / 0.594 |

- **Distractors:** the same doodles drawn again without the things beside the cat give the same 11 answers for
  98.7% of them.
- **Known weaknesses:**
  - The spots pattern is never read (0 of 18; spots are about 1% of the synthetic data).
  - Small regions are the hardest: a cream or brown patch on the nose bridge is found 39% and 50% of the time.
  - Mixed coats are read 90% of the time and tortoiseshell / calico 80%.
- **What these numbers do not show:** the best checkpoint is chosen on the same validation set. For v0 the best
  epoch was the last one, so the choice changed nothing. More importantly, these doodles come from the same
  generator the reader was trained on. They say nothing yet about doodles drawn by people.

## The exam: real doodles

The game's colour doodle board has a "save this doodle" button. The form opens filled in with the reader's
prediction; the author corrects any field and saves. Each doodle becomes a folder under `data/exam/` (not in
git): `board.png`, `answer.json` (the author's answer), `prediction.json` (the reader's answer and version),
`edited.json` (which fields the author changed), `strokes.json` and `meta.json`.

The exam is never training data: `catdoodle.exam` is the only code that reads `data/exam/`. Because the form is
prefilled, the score reports the fields the author changed separately from those they confirmed. The share of
changed fields is the error rate the author saw.

## Use

```bash
pip install -e ".[dev]"
python -m catdoodle.train --out runs/v0        # make the doodles, train, write runs/v0/report.json
python -m catdoodle.exam                        # score the predictions saved with each exam doodle
python -m catdoodle.exam --rerun                # read every exam doodle again with the current weights
python scripts/sample_sheet.py sheet.png        # a sheet of synthetic doodles with their answers
pytest
```

```python
from catdoodle.reader import read_attributes
read_attributes(rgb)   # (H, W, 3) uint8 board on its cream paper -> dict, None = unspecified
```

Training and the sample sheet read Sell Anything's photo labels and face points from a sibling checkout
(`../sell-anything/ml`). The weights are looked up in `CATDOODLE_WEIGHTS`, else `runs/v0/reader.pt`; `runs/` and
`data/` are not in git.
