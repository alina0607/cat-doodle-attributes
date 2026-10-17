import pytest

from catdoodle.schema import NO_PATCH, CatAttributes, Coat, Eye, Pattern, Region


def odd_eyed():
    return CatAttributes(coat=Coat.WHITE, pattern=Pattern.SPOTS, eye_left=Eye.YELLOW, eye_right=Eye.BLUE,
                         patches={Region.CROWN: Coat.BROWN, Region.CHEEK_LEFT: Coat.BLACK,
                                  Region.EAR_RIGHT: NO_PATCH})


def test_nothing_given_is_unspecified_everywhere():
    cat = CatAttributes()
    assert cat.coat is None and cat.eye_left is None and set(cat.patches.values()) == {None}
    assert set(cat.to_dict()["patches"]) == {r.value for r in Region}


def test_mirror_swaps_the_eyes_and_the_left_right_regions():
    m = odd_eyed().mirrored()
    assert m.eye_left is Eye.BLUE and m.eye_right is Eye.YELLOW
    assert m.patches[Region.CHEEK_RIGHT] is Coat.BLACK and m.patches[Region.CHEEK_LEFT] is None
    assert m.patches[Region.EAR_LEFT] == NO_PATCH and m.patches[Region.CROWN] is Coat.BROWN


def test_mirror_twice_and_json_round_trip_give_the_same_cat():
    cat = odd_eyed()
    assert cat.mirrored().mirrored() == cat
    assert CatAttributes.from_json(cat.to_json()) == cat


def test_dict_uses_the_servers_keys():
    d = odd_eyed().to_dict()
    assert d["left_eye"] == "yellow" and d["right_eye"] == "blue"
    assert d["patches"]["crown"] == "brown" and d["patches"]["ear_right"] == "none" and d["patches"]["bridge"] is None


def test_regions_follow_the_landmark_models_left_right_convention():
    landmarks = pytest.importorskip("catlandmarks.landmarks")
    names = landmarks.NAMES
    swapped = [names[i] for i in landmarks.FLIP_PERMUTATION]
    assert swapped[names.index("left_eye")] == "right_eye"            # the same swap as CatAttributes.mirrored


@pytest.mark.parametrize("bad", [dict(coat="purple"), dict(eye_left="red"), dict(patches={"crown": "mixed"}),
                                 dict(patches={"forehead": "black"})])
def test_bad_values_are_errors(bad):
    with pytest.raises(ValueError):
        CatAttributes(**bad)
