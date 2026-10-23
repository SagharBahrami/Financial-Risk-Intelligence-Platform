import numpy as np
import pandas as pd
import pytest

from risk_platform.common.drift import (
    compare_subgroups,
    decompose_rate_change,
    feature_drift,
    ks_statistic,
    psi,
    psi_categorical,
    rate_difference,
    split_half_noise,
    swap_effects,
    wilson_interval,
)


def test_wilson_interval_with_no_events():
    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0, abs=1e-9)
    assert high == pytest.approx(0.2775, abs=1e-3)  # known value for 0 out of 10


def test_wilson_interval_contains_the_rate_and_shrinks_with_n():
    small, large = wilson_interval(50, 1000), wilson_interval(500, 10000)
    assert small[0] < 0.05 < small[1]
    assert large[1] - large[0] < small[1] - small[0]


def test_rate_difference_by_hand():
    # 50/100 against 60/100: se = sqrt(0.25/100 + 0.24/100) = 0.07, z = 0.1 / 0.07
    result = rate_difference(50, 100, 60, 100)
    assert result["diff"] == pytest.approx(0.1)
    assert result["se"] == pytest.approx(0.07)
    assert result["z"] == pytest.approx(0.1 / 0.07)
    assert result["ci_low"] == pytest.approx(0.1 - 1.959964 * 0.07)
    assert result["p_value"] == pytest.approx(0.153, abs=1e-3)


def test_rate_difference_of_equal_rates_is_zero():
    result = rate_difference(30, 300, 100, 1000)
    assert result["diff"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)
    assert result["design_effect_to_lose_significance"] == 1.0  # not significant to begin with


def test_design_effect_inflates_the_standard_error():
    plain, inflated = rate_difference(50, 100, 60, 100), rate_difference(50, 100, 60, 100, design_effect=4.0)
    assert inflated["se"] == pytest.approx(2 * plain["se"])
    assert inflated["diff"] == plain["diff"]


@pytest.mark.parametrize("fp_ref, n_ref, fp_cur, n_cur", [(5859, 117912, 7097, 106718), (5859, 117912, 4280, 95415)])
def test_design_effect_to_lose_significance_is_the_exact_boundary(fp_ref, n_ref, fp_cur, n_cur):
    """Just below the reported design effect the interval excludes zero, just above it includes zero."""
    needed = rate_difference(fp_ref, n_ref, fp_cur, n_cur)["design_effect_to_lose_significance"]
    below = rate_difference(fp_ref, n_ref, fp_cur, n_cur, design_effect=needed * 0.99)
    above = rate_difference(fp_ref, n_ref, fp_cur, n_cur, design_effect=needed * 1.01)
    assert (below["ci_low"] > 0) or (below["ci_high"] < 0)
    assert above["ci_low"] < 0 < above["ci_high"]


def test_design_effect_to_lose_significance_uses_1_96_not_1():
    # z = 5.2 needs (5.2 / 1.96)^2 = 7 times the variance, not 5.2^2 = 27
    needed = rate_difference(5859, 117912, 4280, 95415)["design_effect_to_lose_significance"]
    assert needed == pytest.approx(7, abs=0.5)


def test_psi_categorical_by_hand():
    # shares (0.5, 0.5) -> (0.25, 0.75): 0.25 * ln(0.5/0.25)... = (-0.25) ln(0.5) + 0.25 ln(1.5)
    expected = -0.25 * np.log(0.5) + 0.25 * np.log(1.5)
    assert psi_categorical(["a", "a", "b", "b"], ["a", "b", "b", "b"]) == pytest.approx(expected, rel=1e-5)


def test_psi_is_zero_for_identical_samples_and_grows_with_the_shift():
    values = np.random.default_rng(0).normal(size=5000)
    assert psi(values, values) == pytest.approx(0.0, abs=1e-9)
    small, large = psi(values, values + 0.1), psi(values, values + 1.0)
    assert 0 < small < large
    assert large > 0.25


def test_psi_ignores_nan_and_treats_few_distinct_values_as_categories():
    ref = np.array([0, 1, 1, 0, np.nan] * 50)
    cur = np.array([0, 1, 1, 1, np.nan] * 50)
    assert psi(ref, cur) == pytest.approx(psi_categorical([0, 1, 1, 0] * 50, [0, 1, 1, 1] * 50))


def test_ks_statistic_bounds():
    values = np.arange(100.0)
    assert ks_statistic(values, values) == 0.0
    assert ks_statistic(values, values + 1000) == 1.0


def test_split_half_noise_is_small_for_one_distribution():
    values = np.random.default_rng(1).normal(size=20000)
    noise = split_half_noise(values, psi, n_repeats=30)
    assert noise < 0.01
    assert psi(values, values + 0.2) > 5 * noise  # a real shift is far above the noise level


def test_feature_drift_finds_the_shifted_column_and_the_missing_share():
    rng = np.random.default_rng(2)
    n = 10000

    def frame(shift: float, missing_share: float) -> pd.DataFrame:
        return pd.DataFrame({
            "stable": rng.normal(size=n),
            "shifted": rng.normal(loc=shift, size=n),
            "with_missing": np.where(rng.random(n) < missing_share, -1.0, rng.exponential(size=n)),
            "cat": rng.choice(["a", "b"], size=n, p=[0.5 + shift / 4, 0.5 - shift / 4]),
        })

    table = feature_drift(frame(0, 0.1), frame(1, 0.4), categorical=["cat"], missing_as_negative=["with_missing"])
    assert table["psi"].idxmax() == "shifted"
    assert table.loc["stable", "psi"] < 0.01
    assert table.loc["shifted", "psi"] > 10 * table.loc["shifted", "psi_noise"]
    assert table.loc["with_missing", "missing_ref"] == pytest.approx(0.1, abs=0.02)
    assert table.loc["with_missing", "missing_cur"] == pytest.approx(0.4, abs=0.02)
    assert table.loc["cat", "kind"] == "categorical" and table.loc["cat", "psi"] > 0.05


def test_decompose_rate_change_composition_only():
    # rates stay 0.1 (A) and 0.3 (B); B becomes more common: 50% -> 75%
    ref_groups = ["A"] * 50 + ["B"] * 50
    ref_flags = [1] * 5 + [0] * 45 + [1] * 15 + [0] * 35
    cur_groups = ["A"] * 20 + ["B"] * 60
    cur_flags = [1] * 2 + [0] * 18 + [1] * 18 + [0] * 42
    result = decompose_rate_change(ref_groups, ref_flags, cur_groups, cur_flags)
    assert result["composition"] == pytest.approx(0.05)  # (-0.25 * 0.1) + (0.25 * 0.3)
    assert result["within"] == pytest.approx(0.0)
    assert result["total"] == pytest.approx(0.25 - 0.2)


def test_decompose_rate_change_within_only_and_exact_sum():
    groups = ["A"] * 100 + ["B"] * 100
    ref_flags = [1] * 10 + [0] * 90 + [1] * 30 + [0] * 70
    cur_flags = [1] * 20 + [0] * 80 + [1] * 50 + [0] * 50  # both groups flagged 10 / 20 points more often
    result = decompose_rate_change(groups, ref_flags, groups, cur_flags)
    assert result["composition"] == pytest.approx(0.0)
    assert result["within"] == pytest.approx(0.15)
    assert result["total"] == pytest.approx(np.mean(cur_flags) - np.mean(ref_flags))


def test_decompose_rate_change_with_a_group_missing_in_one_period():
    ref_groups, ref_flags = ["A"] * 100, [1] * 10 + [0] * 90
    cur_groups, cur_flags = ["A"] * 50 + ["B"] * 50, [1] * 5 + [0] * 45 + [1] * 25 + [0] * 25
    result = decompose_rate_change(ref_groups, ref_flags, cur_groups, cur_flags)
    assert result["total"] == pytest.approx(np.mean(cur_flags) - np.mean(ref_flags))
    assert result["composition"] + result["within"] == pytest.approx(result["total"])


def test_compare_subgroups_marks_only_the_changed_group_and_skips_small_groups():
    rng = np.random.default_rng(3)
    n = 20000
    ref_groups = rng.choice(["A", "B", "tiny"], size=n, p=[0.5, 0.495, 0.005])
    cur_groups = rng.choice(["A", "B", "tiny"], size=n, p=[0.5, 0.495, 0.005])
    ref_flags = rng.random(n) < 0.05
    cur_flags = rng.random(n) < np.where(cur_groups == "A", 0.09, 0.05)  # only A changes
    table = compare_subgroups(ref_groups, ref_flags.astype(int), cur_groups, cur_flags.astype(int))
    assert bool(table.loc["A", "significant"])
    assert not bool(table.loc["B", "significant"])
    assert table.loc["A", "diff"] == pytest.approx(0.04, abs=0.01)
    assert np.isnan(table.loc["tiny", "p_value"]) and not bool(table.loc["tiny", "significant"])  # fewer than 200 rows
    assert bool(table.loc["A", "differs_from_rest"])
    assert table.loc["A", "diff_vs_rest"] > 0 > table.loc["B", "diff_vs_rest"]  # A rose more, B less than the rest


def test_uniform_shift_moves_every_subgroup_but_none_differs_from_the_rest():
    rng = np.random.default_rng(5)
    n = 40000
    ref_groups = rng.choice(["A", "B", "C"], size=n)
    cur_groups = rng.choice(["A", "B", "C"], size=n)
    ref_flags = (rng.random(n) < 0.05).astype(int)
    cur_flags = (rng.random(n) < 0.08).astype(int)  # the same increase everywhere
    table = compare_subgroups(ref_groups, ref_flags, cur_groups, cur_flags)
    assert table["significant"].all()
    assert not table["differs_from_rest"].any()
    assert table["diff_vs_rest"].abs().max() < 0.01


def test_diff_vs_rest_by_hand():
    # A: 10% -> 30%; the rest (B): 10% -> 10%, equal sizes, so diff_vs_rest = 0.2 - 0
    groups = ["A"] * 1000 + ["B"] * 1000
    ref_flags = ([1] * 100 + [0] * 900) * 2
    cur_flags = [1] * 300 + [0] * 700 + [1] * 100 + [0] * 900
    table = compare_subgroups(groups, ref_flags, groups, cur_flags)
    assert table.loc["A", "diff_vs_rest"] == pytest.approx(0.2)
    assert table.loc["B", "diff_vs_rest"] == pytest.approx(-0.2)
    assert table.loc["A", "p_vs_rest"] < 1e-10


def test_swap_effects_positive_for_the_driving_column_and_zero_for_others():
    rng = np.random.default_rng(4)
    n = 20000
    current = pd.DataFrame({"driver": rng.normal(loc=1.0, size=n), "other": rng.normal(size=n)})
    reference = pd.DataFrame({"driver": rng.normal(loc=0.0, size=n), "other": rng.normal(size=n)})

    def flag(df):
        return (df["driver"] > 1.0).to_numpy()  # depends on driver only

    effects = swap_effects(flag, current, reference, {"driver": ["driver"], "other": ["other"]})
    assert effects.loc["driver", "mean"] == pytest.approx(0.5 - 0.1587, abs=0.02)  # P(N(1,1) > 1) - P(N(0,1) > 1)
    assert effects.loc["other", "mean"] == pytest.approx(0.0, abs=0.02)


def test_wilson_interval_design_effect_widens_the_interval_and_keeps_the_rate():
    plain, inflated = wilson_interval(500, 10000), wilson_interval(500, 10000, design_effect=4.0)
    assert inflated[0] < plain[0] < 0.05 < plain[1] < inflated[1]
    assert (inflated[1] - inflated[0]) == pytest.approx(2 * (plain[1] - plain[0]), rel=0.05)  # sqrt(4) = 2
    assert (inflated[0] + inflated[1]) / 2 == pytest.approx(0.05, abs=0.002)  # still centred near the rate
    assert wilson_interval(500, 10000, design_effect=1.0) == plain


def test_wilson_interval_matches_rate_difference_scaling():
    """Both use the variance multiplied by the design effect, so the widths scale the same way."""
    d = 9.0
    wilson_ratio = np.subtract(*wilson_interval(300, 5000, design_effect=d)[::-1]) / np.subtract(*wilson_interval(300, 5000)[::-1])
    wald_ratio = rate_difference(300, 5000, 400, 5000, d)["se"] / rate_difference(300, 5000, 400, 5000)["se"]
    assert wilson_ratio == pytest.approx(wald_ratio, rel=0.05)  # both about sqrt(9) = 3


def _swap_frames():
    rng = np.random.default_rng(6)
    n = 500
    a = rng.normal(size=n)
    current = pd.DataFrame({"a": a, "b": -a, "c": rng.normal(size=n)})            # b = -a in the current month
    ref_a = rng.normal(loc=2.0, size=n)
    reference = pd.DataFrame({"a": ref_a, "b": 10 * ref_a, "c": rng.normal(size=n)})  # b = 10 a in the reference
    return current, reference


def test_swap_effects_keeps_the_relationship_inside_a_group():
    current, reference = _swap_frames()
    seen = []

    def flag(df):
        seen.append(df.copy())
        return np.zeros(len(df))

    swap_effects(flag, current, reference, {"ab": ["a", "b"]}, n_repeats=3)
    own_shuffled, from_reference = seen[0], seen[1]   # the order inside one repetition: control, then reference draw
    # reference draw: a and b come from the same reference rows, so b = 10 a still holds
    assert np.allclose(from_reference["b"], 10 * from_reference["a"])
    assert set(from_reference["a"]).issubset(set(reference["a"]))
    # control: a and b are shuffled together, so b = -a still holds, and the values are the current ones
    assert np.allclose(own_shuffled["b"], -own_shuffled["a"])
    assert sorted(own_shuffled["a"]) == pytest.approx(sorted(current["a"]))
    # the column outside the group is untouched
    assert from_reference["c"].equals(current["c"]) and own_shuffled["c"].equals(current["c"])


def test_swap_effects_is_reproducible_and_reports_the_spread_of_five_repetitions():
    current, reference = _swap_frames()
    flag = lambda df: (df["a"] > 1.5).to_numpy()  # noqa: E731
    groups = {"a": ["a"], "ab": ["a", "b"], "c": ["c"]}
    first = swap_effects(flag, current, reference, groups)             # default: 5 repetitions
    second = swap_effects(flag, current, reference, groups)
    pd.testing.assert_frame_equal(first, second)                         # same seeds, same result
    assert list(first.columns) == ["mean", "std"]
    assert (first.loc[["a", "ab"], "std"] > 0).all()                     # the draws do differ
    assert first.loc["c", "mean"] == 0.0                                 # the flag rule ignores column c
    other_seed = swap_effects(flag, current, reference, groups, seed=100)
    assert not np.allclose(first["mean"], other_seed["mean"])
    calls = []
    swap_effects(lambda df: calls.append(1) or np.zeros(len(df)), current, reference, {"a": ["a"]})
    assert len(calls) == 2 * 5                                           # 5 repetitions x (reference draw + control)
    assert np.isnan(swap_effects(flag, current, reference, {"a": ["a"]}, n_repeats=1).loc["a", "std"])
