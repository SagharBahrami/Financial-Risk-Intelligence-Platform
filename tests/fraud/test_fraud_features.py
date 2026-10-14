import pandas as pd
import pytest

from risk_platform.fraud.features import add_missing_flags, time_split


def test_add_missing_flags():
    df = pd.DataFrame({"a": [-1, 5, 0], "b": [2.0, -0.5, 3.0]})
    result = add_missing_flags(df, ["a", "b"])

    assert result["a_missing"].tolist() == [1, 0, 0]
    assert result["b_missing"].tolist() == [0, 1, 0]
    assert result["a"].isna().tolist() == [True, False, False]
    assert result["b"].isna().tolist() == [False, True, False]
    assert df["a"].tolist() == [-1, 5, 0]  # input is not modified


def test_time_split_keeps_periods_separate(interim_baf):
    train, test = time_split(interim_baf, "month", [0, 1, 2, 3, 4, 5], [6, 7])

    assert set(train["month"]) <= {0, 1, 2, 3, 4, 5}
    assert set(test["month"]) <= {6, 7}
    assert len(train) + len(test) == len(interim_baf)


def test_time_split_rejects_overlap(interim_baf):
    with pytest.raises(ValueError):
        time_split(interim_baf, "month", [0, 1, 2], [2, 3])
