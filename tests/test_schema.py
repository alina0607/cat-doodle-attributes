import numpy as np
import pytest

from catdoodle.schema import SHAPE_GRID, CatAttributes, Coat, Patch, Pattern, Region

WHITE, BROWN, YELLOW, BLUE = (95.0, 0.0, 2.0), (35.0, 12.0, 25.0), (80.0, 5.0, 70.0), (60.0, -5.0, -40.0)


def heart():
    """A small asymmetric mask, so a wrong flip shows up."""
    shape = np.zeros((SHAPE_GRID, SHAPE_GRID), bool)
    shape[4:8, 2:6] = True
    shape[6:12, 5:9] = True
    return shape


def odd_eyed(patches=None):
    return CatAttributes(coat=Coat.WHITE, coat_colour=WHITE, pattern=Pattern.PATCHED, left_eye=YELLOW, right_eye=BLUE,
                         patches=patches or {Region.CROWN: Patch(colour=BROWN, coverage=0.3, shape=heart())})


def test_mirror_swaps_the_eyes_and_the_left_right_regions():
    cat = odd_eyed({Region.LEFT_CHEEK: Patch(colour=BROWN, coverage=0.2),
                    Region.RIGHT_EAR: Patch(colour=BROWN, coverage=0.5)})
    m = cat.mirrored()
    assert m.left_eye == BLUE and m.right_eye == YELLOW
    assert set(m.patches) == {Region.RIGHT_CHEEK, Region.LEFT_EAR}


def test_mirror_flips_a_patch_shape_and_twice_is_the_identity():
    cat = odd_eyed()
    m = cat.mirrored()
    assert (m.patches[Region.CROWN].shape == heart()[:, ::-1]).all()
    back = m.mirrored()
    assert back == cat and (back.patches[Region.CROWN].shape == heart()).all()


def test_json_round_trip_keeps_everything():
    cat = odd_eyed()
    again = CatAttributes.from_json(cat.to_json())
    assert again == cat and (again.patches[Region.CROWN].shape == heart()).all()


def test_regions_follow_the_landmark_models_left_right_convention():
    landmarks = pytest.importorskip("catlandmarks.landmarks")
    names = landmarks.NAMES
    swapped = [names[i] for i in landmarks.FLIP_PERMUTATION]
    assert swapped[names.index("left_eye")] == "right_eye"            # the same swap as CatAttributes.mirrored
    assert {r.value for r in Region} >= {"left_cheek", "right_cheek", "left_ear", "right_ear"}


@pytest.mark.parametrize("bad", [dict(coverage=1.5), dict(colour=(120.0, 0.0, 0.0)),
                                 dict(shape=np.zeros((8, 8), bool))])
def test_bad_values_are_errors(bad):
    with pytest.raises(ValueError):
        Patch(**{"colour": BROWN, "coverage": 0.3, **bad})
