# Cat Doodle Attributes

> A model that reads a coloured cat doodle and reports what the cat looks like: its coat colour, which parts of
> the face carry a patch or stripes and in what colour, and the colour of each eye. Its output is the condition
> for a generative model that paints the same cat in oil, with its own light and shade.

**Status: design.** No model has been trained yet. This README states what will be built and how it will be
judged; numbers will be added only when they come from a run.

## Where it fits

This model serves [Sell Anything](https://github.com/alina0607/sell-anything), a game whose cats are painted
stroke by stroke by an autoregressive transformer (Brush GPT). In a planned level the player doodles a cat,
and the game paints an oil cat that follows the doodle's *description*, not its pixels:

```
player's doodle
  └─► is it a cat?                       sketch-recognition-from-scratch  (sketch classifier)
        └─► what does the cat look like?  cat-doodle-attributes            (this repository)
              └─► paint that cat           sell-anything                    (Brush GPT, conditioned on the attributes)
```

A white cat with a brown patch on the crown, a yellow left eye and a blue right eye should come out as an oil
painting of a white cat with a brown crown patch and those two eye colours — wherever the player drew it and
however large. The painting's shading and fur direction are the painting model's own; the doodle only says
*what*, never *how*.

## What it reads

| Attribute | Form | Example |
|---|---|---|
| Coat colour | a category and a colour | white |
| Pattern | solid, tabby (stripes), spotted / patched, bicolour, tortoiseshell / calico | solid |
| Per face region: patch or not, its colour, its rough shape | regions: crown, left cheek, right cheek, nose bridge, muzzle and chin, left ear, right ear | crown: brown patch |
| Left eye, right eye | a colour each | yellow, blue |
| The nine face points | positions in the doodle | — |

**Left and right are as seen in the image**, the convention of
[cat-face-landmarks](https://github.com/alina0607/cat-face-landmarks): the left eye is the one on the left of the
picture. The face points, the regions and the eyes all use that one convention, and a mirror image swaps them
with the same permutation the landmark model uses.

Colours drawn **outside** the cat's face — a coloured background, a ball next to the cat, scribbles — are not
the cat and must not change the reading. A stroke that slightly overshoots the outline still belongs to the cat.

## How it will be built

1. **Labels from photos.** For every AFHQ cat, the coat colour, the region patches and each eye's colour are
   measured on the photo, inside the subject mask, with the regions placed by the cat's nine face points.
2. **Doodles from photos.** Each photo is turned into a doodle-like picture: outlines plus flat colour fills
   (as Sell Anything's flat-colour cat faces), roughened like a hand drawing — wobbly lines, unfilled gaps,
   overshoots — and given distractors outside the face whose colours the labels ignore.
3. **The reader.** A vision model trained on those pairs to output the attributes and the face points; its
   encoder may start from cat-face-landmarks' self-supervised encoder.
4. **The judge is real doodles.** Synthetic doodles are never the test. A held-out set of cat doodles drawn by
   people, each with the attributes its author intended, is used only for evaluation — including doodles with
   colour outside the face, to check the reading does not move.

Nothing in the reading is a hand-written rule standing in for the model: the attributes come from the model,
trained on data, as the Sell Anything project requires.
