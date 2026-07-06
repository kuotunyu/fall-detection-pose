"""tune/test 切分邏輯的測試(純 Python,不碰網路/資料)。"""

from fall_detection.eval.splits import generate_splits, load_splits, save_splits, split_of
from fall_detection.io.urfd import adl_sequences, fall_sequences


def test_split_counts_match_agreed_ratio():
    splits = generate_splits()
    assert len(splits["tune"]["falls"]) == 10
    assert len(splits["tune"]["adls"]) == 13
    assert len(splits["test"]["falls"]) == 20
    assert len(splits["test"]["adls"]) == 27


def test_splits_cover_every_sequence_without_overlap():
    splits = generate_splits()
    tune_falls, test_falls = set(splits["tune"]["falls"]), set(splits["test"]["falls"])
    tune_adls, test_adls = set(splits["tune"]["adls"]), set(splits["test"]["adls"])
    assert tune_falls & test_falls == set()  # tune/test 不重疊
    assert tune_adls & test_adls == set()
    assert tune_falls | test_falls == set(fall_sequences())  # 70 支一支不漏
    assert tune_adls | test_adls == set(adl_sequences())


def test_generate_splits_is_deterministic():
    assert generate_splits(seed=42) == generate_splits(seed=42)


def test_different_seed_gives_different_split():
    assert generate_splits(seed=42) != generate_splits(seed=1)


def test_save_load_roundtrip(tmp_path):
    splits = generate_splits()
    path = tmp_path / "splits.yaml"
    save_splits(splits, path)
    assert load_splits(path) == splits


def test_split_of_reports_correct_membership():
    splits = generate_splits()
    for seq in splits["tune"]["falls"] + splits["tune"]["adls"]:
        assert split_of(seq, splits) == "tune"
    for seq in splits["test"]["falls"] + splits["test"]["adls"]:
        assert split_of(seq, splits) == "test"


def test_split_of_unknown_sequence_raises():
    splits = generate_splits()
    try:
        split_of("fall-99", splits)
    except KeyError:
        pass
    else:
        raise AssertionError("fall-99 不在任何名單中,應該要拋 KeyError")
