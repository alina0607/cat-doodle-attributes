import json

from catdoodle.exam import load_exam, score
from catdoodle.schema import CatAttributes


def _save(folder, name, answer, prediction, edited):
    d = folder / name
    d.mkdir(parents=True)
    (d / "answer.json").write_text(json.dumps(answer.to_dict()))
    (d / "prediction.json").write_text(json.dumps({"attributes": prediction.to_dict(), "reader": "v0"}))
    (d / "edited.json").write_text(json.dumps(edited))


def test_scores_the_saved_predictions_and_counts_what_the_author_changed(tmp_path):
    white = CatAttributes(coat="white", eye_left="yellow", eye_right="blue", patches={"crown": "brown"})
    _save(tmp_path, "a", white, white, {"coat": False})
    _save(tmp_path, "b", white, CatAttributes(coat="cream", eye_left="yellow"),
          {"coat": True, "right_eye": True, "patches": {"crown": True}})
    items = load_exam(tmp_path)
    assert [i["id"] for i in items] == ["a", "b"]
    s = score(items)
    assert s["coat"]["right"] == 1 and s["coat"]["total"] == 2 and s["coat"]["changed"] == 1
    assert s["coat"]["right_confirmed"] == 1 and s["coat"]["confirmed"] == 1
    assert s["left_eye"]["accuracy"] == 1.0 and s["right_eye"]["accuracy"] == 0.5
    assert s["patches.crown"]["accuracy"] == 0.5 and s["patches.bridge"]["accuracy"] == 1.0


def test_an_empty_or_missing_exam_is_no_doodles(tmp_path):
    assert load_exam(tmp_path) == [] and load_exam(tmp_path / "nothing") == []
