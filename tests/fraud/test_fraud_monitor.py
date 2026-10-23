import numpy as np
import pandas as pd
import pytest

from risk_platform.fraud.monitor import (
    build_report,
    fpr_differences,
    input_drift_tables,
    performance_table,
    render_report,
    score_drift_table,
    subgroup_columns,
)

CONFIG = {
    "target": "fraud_bool",
    "time_column": "month",
    "categorical_columns": ["kind"],
    "features": {"missing_as_negative": ["gap"]},
    "evaluation": {"max_fpr": 0.25},
}


def test_performance_table_by_hand():
    scored = pd.DataFrame({
        "month": 0,
        "fraud_bool": [0, 0, 0, 0, 1, 1],
        "score": [0.1, 0.2, 0.6, 0.7, 0.8, 0.3],
    })
    row = performance_table(scored, threshold=0.5, months=[0], config=CONFIG).loc[0]
    # flagged: 0.6, 0.7 (false positives) and 0.8 (true positive)
    assert (row["n"], row["n_fraud"], row["n_legit"]) == (6, 2, 4)
    assert row["fraud_rate"] == pytest.approx(2 / 6)
    assert row["flag_rate"] == pytest.approx(0.5)
    assert row["fpr"] == pytest.approx(0.5)
    assert row["recall"] == pytest.approx(0.5)
    assert row["precision"] == pytest.approx(1 / 3)
    assert row["fpr_low"] < 0.5 < row["fpr_high"]
    assert row["roc_auc"] == pytest.approx(0.75)  # 6 of the 8 fraud/legit pairs are ranked correctly
    # best threshold with FPR <= 0.25 keeps one of four legit cases flagged: only 0.8 and 0.7 -> 0.8 first
    assert row["recall_at_max_fpr"] == pytest.approx(0.5)


def test_threshold_is_inclusive():
    scored = pd.DataFrame({"month": 0, "fraud_bool": [0, 1], "score": [0.5, 0.5]})
    row = performance_table(scored, threshold=0.5, months=[0], config=CONFIG).loc[0]
    assert row["fpr"] == 1.0 and row["recall"] == 1.0  # a score equal to the threshold is flagged


def make_months(shift_in_month_2: float = 0.0, n: int = 6000, seed: int = 0) -> pd.DataFrame:
    """Two months of data. The score is a function of x; month 2 has x shifted up by `shift_in_month_2`."""
    rng = np.random.default_rng(seed)
    frames = []
    for month, shift in [(1, 0.0), (2, shift_in_month_2)]:
        y = (rng.random(n) < 0.05).astype(int)
        x = rng.normal(loc=y * 1.5 + shift, size=n)
        frames.append(pd.DataFrame({
            "month": month, "fraud_bool": y, "x": x,
            "stable": rng.normal(size=n),
            "gap": np.where(rng.random(n) < 0.2, -1.0, rng.normal(size=n)),
            "kind": rng.choice(["a", "b"], size=n),
            "flag_yes_no": rng.integers(0, 2, size=n),
        }))
    df = pd.concat(frames, ignore_index=True)
    return df.assign(score=1 / (1 + np.exp(-(df["x"] - 3))))


def test_score_drift_table_detects_a_shift_above_the_noise_level():
    scored = make_months(shift_in_month_2=1.0)
    threshold = scored.loc[(scored.month == 1) & (scored.fraud_bool == 0), "score"].quantile(0.95)
    table, noise = score_drift_table(scored, threshold, reference=1, months=[2], config=CONFIG, noise_repeats=20,
                                     legit_only=True)
    assert table.loc[1, "share_above_threshold"] == pytest.approx(0.05, abs=0.001)
    assert table.loc[1, "threshold_percentile"] == pytest.approx(95, abs=0.1)
    assert table.loc[2, "share_above_threshold"] > 0.10
    assert table.loc[2, "psi"] > 10 * noise["psi"]
    assert table.loc[2, "ks"] > 5 * noise["ks"]


def test_no_shift_stays_within_noise():
    scored = make_months(shift_in_month_2=0.0)
    threshold = scored.loc[(scored.month == 1) & (scored.fraud_bool == 0), "score"].quantile(0.95)
    table, noise = score_drift_table(scored, threshold, reference=1, months=[2], config=CONFIG, noise_repeats=20,
                                     legit_only=True)
    assert table.loc[2, "psi"] < 0.01
    assert abs(table.loc[2, "share_above_threshold"] - 0.05) < 0.02


def test_fpr_differences_labels_and_sign():
    performance = pd.DataFrame(
        {"false_pos": [50, 80], "n_legit": [1000, 1000]}, index=[1, 2]
    )
    table = fpr_differences(performance)
    assert list(table.index) == ["2 - 1"]
    assert table.loc["2 - 1", "diff"] == pytest.approx(0.03)


def test_subgroup_columns_include_categoricals_and_yes_no_features_only():
    columns = subgroup_columns(make_months(), CONFIG)
    assert set(columns) == {"kind", "flag_yes_no"}


def test_build_report_separates_the_sections_and_finds_the_shifted_feature():
    scored = make_months(shift_in_month_2=1.0)
    threshold = scored.loc[(scored.month == 1) & (scored.fraud_bool == 0), "score"].quantile(0.95)
    flag_fn = lambda Z: (1 / (1 + np.exp(-(Z["x"] - 3)))) >= threshold  # the same rule as the score  # noqa: E731
    report = build_report(scored, threshold, reference=1, months=[2], config=CONFIG, flag_fn=flag_fn)

    # performance change
    assert report["performance"].loc[2, "fpr"] > report["performance"].loc[1, "fpr"] + 0.05
    assert report["fpr_differences"].loc["2 - 1", "ci_low"] > 0
    # input drift: x is the shifted feature
    drift = report["input_drift"][2]
    assert drift["psi"].idxmax() == "x"
    assert drift.loc["stable", "psi"] < 0.01
    # the extra false positives are within-group, not a change of the mix
    splits = report["subgroup_splits"].loc[(2, "kind")]
    assert splits["within"] > 10 * abs(splits["composition"])
    assert splits["total"] == pytest.approx(splits["composition"] + splits["within"])
    # the swap diagnostic points to x
    assert report["swap"][(2, "mean")].idxmax() == "x"
    assert report["swap"].loc["stable", (2, "mean")] == pytest.approx(0.0, abs=1.0)  # percentage points
    assert (report["swap"][(2, "std")] >= 0).all()


def test_render_report_has_the_three_kinds_of_change_and_the_caveats():
    scored = make_months(shift_in_month_2=1.0)
    threshold = scored.loc[(scored.month == 1) & (scored.fraud_bool == 0), "score"].quantile(0.95)
    flag_fn = lambda Z: (1 / (1 + np.exp(-(Z["x"] - 3)))) >= threshold  # noqa: E731
    text = render_report(build_report(scored, threshold, 1, [2], CONFIG, flag_fn=flag_fn), calibration_months=[1])
    for heading in ("## 1. Performance", "## 2a. Score drift of ALL applications (no labels needed)",
                    "## 2b. Score drift of LEGITIMATE applications (needs labels)",
                    "## 3. Input drift (all applications", "## 4. Subgroups", "## 5. Feature-swap"):
        assert heading in text
    assert "needs labels" in text and "no labels needed" in text
    assert "not an independent benchmark" in text
    assert "not evidence of what caused the change" in text
    assert "independent rows" in text


def _threshold(scored: pd.DataFrame) -> float:
    return scored.loc[(scored.month == 1) & (scored.fraud_bool == 0), "score"].quantile(0.95)


def test_all_application_monitoring_needs_no_labels():
    """Score drift of all applications and input drift work on a frame without the target column."""
    scored = make_months(shift_in_month_2=1.0)
    threshold = _threshold(scored)
    unlabeled = scored.drop(columns=["fraud_bool"])

    table, noise = score_drift_table(unlabeled, threshold, reference=1, months=[2], config=CONFIG, noise_repeats=10)
    labeled_table, _ = score_drift_table(scored, threshold, reference=1, months=[2], config=CONFIG, noise_repeats=10)
    pd.testing.assert_frame_equal(table, labeled_table)          # the labels play no role in this version
    assert "fraud_median" not in table.columns                   # that column would need the labels
    assert table.loc[2, "psi"] > 10 * noise["psi"]

    drift = input_drift_tables(unlabeled, reference=1, months=[2], config=CONFIG)[2]
    assert drift["psi"].idxmax() == "x"


def test_all_application_and_legitimate_only_scores_are_different_populations():
    scored = make_months(shift_in_month_2=0.0)
    threshold = _threshold(scored)
    everyone, _ = score_drift_table(scored, threshold, 1, [2], CONFIG, noise_repeats=10)
    legit, _ = score_drift_table(scored, threshold, 1, [2], CONFIG, noise_repeats=10, legit_only=True)
    month_1 = scored[scored.month == 1]
    # all applications: the share above the threshold is the flag rate (fraud included)
    assert everyone.loc[1, "share_above_threshold"] == pytest.approx((month_1.score >= threshold).mean())
    # legitimate only: it is the FPR, and the fraud median is added (needs the labels)
    fpr = (month_1.loc[month_1.fraud_bool == 0, "score"] >= threshold).mean()
    assert legit.loc[1, "share_above_threshold"] == pytest.approx(fpr)
    assert "fraud_median" in legit.columns
    assert everyone.loc[1, "share_above_threshold"] > legit.loc[1, "share_above_threshold"]  # fraud scores are higher


def test_design_effect_widens_the_monthly_intervals_too():
    scored = make_months(shift_in_month_2=1.0)
    threshold = _threshold(scored)
    plain = performance_table(scored, threshold, [1, 2], CONFIG)
    inflated = performance_table(scored, threshold, [1, 2], CONFIG, design_effect=4.0)
    assert (inflated["fpr"] == plain["fpr"]).all() and (inflated["recall"] == plain["recall"]).all()
    for column in ("fpr", "recall"):
        plain_width = plain[f"{column}_high"] - plain[f"{column}_low"]
        inflated_width = inflated[f"{column}_high"] - inflated[f"{column}_low"]
        assert np.allclose(inflated_width / plain_width, 2.0, rtol=0.05)  # sqrt(4)

    # and in the whole report, where it also widens the FPR differences
    plain_report = build_report(scored, threshold, 1, [2], CONFIG)
    wide_report = build_report(scored, threshold, 1, [2], CONFIG, design_effect=4.0)
    width = lambda r: r["fpr_differences"].loc["2 - 1", "ci_high"] - r["fpr_differences"].loc["2 - 1", "ci_low"]  # noqa: E731
    assert width(wide_report) == pytest.approx(2 * width(plain_report), rel=1e-6)
    assert (wide_report["fpr_differences"]["design_effect_to_lose_significance"]
            == plain_report["fpr_differences"]["design_effect_to_lose_significance"]).all()  # defined from independent rows


def test_swap_table_is_reproducible():
    scored = make_months(shift_in_month_2=1.0)
    threshold = _threshold(scored)
    flag_fn = lambda Z: (1 / (1 + np.exp(-(Z["x"] - 3)))) >= threshold  # noqa: E731
    first = build_report(scored, threshold, 1, [2], CONFIG, flag_fn=flag_fn)["swap"]
    second = build_report(scored, threshold, 1, [2], CONFIG, flag_fn=flag_fn)["swap"]
    pd.testing.assert_frame_equal(first, second)
