import numpy as np

from catdoodle.model import decode, encode
from catdoodle.schema import NO_PATCH, Coat, Region
from catdoodle.synth import MEDIAN_POINTS, PALETTE, PAPER, SIDE, draw_doodle, region_masks

RECORD = {"base_colour": "white", "pattern": "solid", "eye_left": {"name": "yellow"},
          "eye_right": {"name": "blue"}, "regions": {"crown": {"has_patch": True, "patch_name": "brown"}}}


def test_a_doodle_is_a_picture_with_an_answer_that_survives_encoding():
    for seed in range(20):
        d = draw_doodle(np.random.default_rng(seed), RECORD, MEDIAN_POINTS)
        assert d.image.shape == (SIDE, SIDE, 3) and d.image.dtype == np.uint8
        assert decode(encode(d.answer)) == d.answer
        assert d.points.shape == (9, 2)


def test_the_answer_is_what_was_drawn():
    seen_patch = seen_unpainted = False
    for seed in range(60):
        a = draw_doodle(np.random.default_rng(seed), RECORD, MEDIAN_POINTS).answer
        if a.coat is None:                          # only an outline: nothing about the fur is known
            seen_unpainted = True
            assert a.pattern is None and set(a.patches.values()) == {None}
        else:
            assert a.coat is Coat.WHITE and None not in a.patches.values()
            seen_patch |= a.patches[Region.CROWN] is Coat.BROWN
        painted = {None, "yellow", "green", "blue", "orange"}            # copper has no paint in the palette
        assert a.eye_left in painted and a.eye_right in painted
    assert seen_patch and seen_unpainted


def test_distractors_only_add_things_beside_the_cat():
    for seed in range(30):
        with_ = draw_doodle(np.random.default_rng(seed), RECORD, MEDIAN_POINTS)
        without = draw_doodle(np.random.default_rng(seed), RECORD, MEDIAN_POINTS, distractors=False)
        assert with_.answer == without.answer


def test_the_seven_regions_do_not_overlap():
    u, v = np.meshgrid(np.linspace(-1.6, 1.6, 200), np.linspace(-1.4, 2.0, 200))
    masks = region_masks(u, v, MEDIAN_POINTS)
    assert set(masks) == set(Region)
    assert (sum(m.astype(int) for m in masks.values()) <= 1).all()


def test_paper_is_not_white():
    assert PAPER != PALETTE["white"] and NO_PATCH == "none"
