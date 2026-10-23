"""Drift and monitoring calculations, shared by all use cases.

Three different things can change between two periods, and they are kept apart on purpose:
- input drift         the distribution of the features changed (psi, ks_statistic, feature_drift)
- score drift         the distribution of the model scores changed, so a fixed threshold flags a different
                      share of cases (psi and ks_statistic on the scores)
- performance change  the decisions or the ranking of the model got better or worse; this needs labels
                      (rate_difference on FPR or recall, compare_subgroups)

What needs labels: input drift and score drift of ALL applications do not, so they can be monitored in
production. Anything restricted to one class (for example the scores of legitimate applications, whose share
above the threshold is the FPR) needs the labels, so it can only be computed once the outcomes are known.

Intervals and tests treat rows as independent. Data with repeated customers or bursts in time has more
variance than that, so the intervals are too narrow. Pass design_effect > 1 to inflate the variance of every
interval and test that is a probability statement: wilson_interval, rate_difference, compare_subgroups.
psi, ks_statistic and split_half_noise are not intervals and ignore it. Use
rate_difference()["design_effect_to_lose_significance"] to see how much inflation a result can survive.

swap_effects() is a diagnostic: it shows how much a feature's values are associated with a change of the flag
rate when the model is applied to modified data. It does not show that the feature caused the change.
"""
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

Z_95 = 1.959964  # two-sided 95% normal quantile


def wilson_interval(successes: int, n: int, z: float = Z_95, design_effect: float = 1.0) -> tuple[float, float]:
    """Wilson score interval for a proportion (better than the normal interval for small or extreme rates).

    design_effect > 1 widens the interval for clustered rows: the same rate is treated as if it was measured on
    n / design_effect independent rows.
    """
    p = successes / n
    n = n / design_effect
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return float(centre - half), float(centre + half)


def rate_difference(
    successes_ref: int, n_ref: int, successes_cur: int, n_cur: int, design_effect: float = 1.0
) -> dict[str, float]:
    """Difference of two proportions (current - reference) with a 95% interval and a z-test.

    design_effect multiplies the variance (1 = independent rows). design_effect_to_lose_significance is the
    design effect at which the interval would just reach zero: (|z| / 1.96)^2, with z from independent rows.
    1.0 means the difference is not significant even for independent rows.
    """
    p_ref, p_cur = successes_ref / n_ref, successes_cur / n_cur
    diff = p_cur - p_ref
    naive_se = np.sqrt(p_ref * (1 - p_ref) / n_ref + p_cur * (1 - p_cur) / n_cur)
    se = naive_se * np.sqrt(design_effect)
    z_naive = diff / naive_se if naive_se > 0 else 0.0
    return {
        "diff": float(diff),
        "se": float(se),
        "ci_low": float(diff - Z_95 * se),
        "ci_high": float(diff + Z_95 * se),
        "z": float(z_naive / np.sqrt(design_effect)),
        "p_value": float(2 * stats.norm.sf(abs(z_naive) / np.sqrt(design_effect))),
        "design_effect_to_lose_significance": float(max(1.0, (abs(z_naive) / Z_95) ** 2)),
    }


def _clean(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[~np.isnan(values)]


def psi_categorical(reference, current, eps: float = 1e-6) -> float:
    """Population stability index between the category frequencies of two samples."""
    reference, current = pd.Series(reference), pd.Series(current)
    categories = sorted(set(reference.dropna()) | set(current.dropna()), key=str)
    p_ref = reference.value_counts(normalize=True).reindex(categories).fillna(0).clip(lower=eps).to_numpy()
    p_cur = current.value_counts(normalize=True).reindex(categories).fillna(0).clip(lower=eps).to_numpy()
    return float(((p_cur - p_ref) * np.log(p_cur / p_ref)).sum())


def psi(reference, current, bins: int = 10, eps: float = 1e-6) -> float:
    """Population stability index of a numeric variable, with bins from the quantiles of the reference.

    Rule of thumb: below 0.1 small, 0.1-0.25 moderate, above 0.25 large. NaN values are ignored.
    A variable with at most `bins` distinct values in the reference is treated as categorical.
    """
    ref, cur = _clean(reference), _clean(current)
    if len(np.unique(ref)) <= bins:
        return psi_categorical(ref, cur, eps)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    p_ref = np.clip(np.histogram(ref, edges)[0] / len(ref), eps, None)
    p_cur = np.clip(np.histogram(cur, edges)[0] / len(cur), eps, None)
    return float(((p_cur - p_ref) * np.log(p_cur / p_ref)).sum())


def ks_statistic(reference, current) -> float:
    """Largest distance between the two empirical distribution functions (0 = same, 1 = no overlap)."""
    return float(stats.ks_2samp(_clean(reference), _clean(current)).statistic)


def split_half_noise(
    values, statistic: Callable[[np.ndarray, np.ndarray], float], n_repeats: int = 100, seed: int = 0,
    quantile: float = 0.95,
) -> float:
    """How large `statistic` gets by chance: the 95th percentile between two random halves of one sample.

    A drift value near this level cannot be told apart from sampling noise.
    """
    values = _clean(values)
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_repeats):
        order = rng.permutation(len(values))
        half = len(values) // 2
        results.append(statistic(values[order[:half]], values[order[half:]]))
    return float(np.quantile(results, quantile))


def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    categorical: list[str],
    missing_as_negative: list[str] = (),
    noise_repeats: int = 20,
) -> pd.DataFrame:
    """Input drift per feature: PSI (and KS for numbers), the PSI noise level, and the share of missing values.

    Columns in `missing_as_negative` use negative numbers for "missing" (as in BAF); these are treated as
    missing, and the change of their missing share is reported separately.
    """
    rows = {}
    for column in reference.columns:
        ref, cur = reference[column], current[column]
        row: dict[str, float | str] = {"kind": "categorical" if column in categorical else "numeric"}
        if column in categorical:
            row["psi"] = psi_categorical(ref, cur)
            row["ks"] = np.nan
            halves = ref.sample(frac=0.5, random_state=0)
            row["psi_noise"] = psi_categorical(halves, ref.drop(halves.index))
            row["missing_ref"] = row["missing_cur"] = np.nan
        else:
            ref, cur = ref.astype(float), cur.astype(float)
            if column in missing_as_negative:
                row["missing_ref"], row["missing_cur"] = float((ref < 0).mean()), float((cur < 0).mean())
                ref, cur = ref.where(ref >= 0), cur.where(cur >= 0)
            else:
                row["missing_ref"], row["missing_cur"] = float(ref.isna().mean()), float(cur.isna().mean())
            row["psi"] = psi(ref, cur)
            row["ks"] = ks_statistic(ref, cur)
            row["psi_noise"] = split_half_noise(ref, psi, n_repeats=noise_repeats)
        rows[column] = row
    return pd.DataFrame(rows).T.astype({"psi": float, "ks": float, "psi_noise": float,
                                        "missing_ref": float, "missing_cur": float})


def _group_stats(groups: pd.Series, flags: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"group": groups.to_numpy(), "flag": flags.to_numpy()}).groupby("group", dropna=False)["flag"].agg(
        n="size", flagged="sum")


def decompose_rate_change(ref_groups, ref_flags, cur_groups, cur_flags) -> dict[str, float]:
    """Split the change of a flag rate between two periods into composition and within-group parts.

    composition: groups became more or less common, at their reference rates
    within:      the rate inside the groups changed, at the current group shares
    composition + within = total (exactly). A group missing in one period takes the rate of the other.
    """
    ref = _group_stats(pd.Series(ref_groups), pd.Series(ref_flags))
    cur = _group_stats(pd.Series(cur_groups), pd.Series(cur_flags))
    groups = ref.index.union(cur.index)
    ref, cur = ref.reindex(groups), cur.reindex(groups)
    w_ref, w_cur = ref["n"].fillna(0) / ref["n"].sum(), cur["n"].fillna(0) / cur["n"].sum()
    r_ref, r_cur = ref["flagged"] / ref["n"], cur["flagged"] / cur["n"]
    r_ref, r_cur = r_ref.fillna(r_cur), r_cur.fillna(r_ref)
    composition = float(((w_cur - w_ref) * r_ref).sum())
    within = float((w_cur * (r_cur - r_ref)).sum())
    return {"total": composition + within, "composition": composition, "within": within}


def compare_subgroups(
    ref_groups, ref_flags, cur_groups, cur_flags, design_effect: float = 1.0, min_count: int = 200,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Flag rate per subgroup in both periods, the difference, and whether it is significant.

    Only subgroups with at least min_count rows in both periods are tested. Both flags use a Bonferroni
    correction (alpha divided by the number of tests), because many subgroups are compared at once.
    - significant:        the flag rate of the subgroup changed
    - differs_from_rest:  it changed differently from the rest of the population (diff_vs_rest = the subgroup's
                          change minus the change of all other rows). When the whole score level moves, almost
                          every subgroup is "significant", so this is the flag that points to specific subgroups.
    These are differences in percentage points. A subgroup with a high flag rate moves more in points than one
    with a low rate for the same relative change, so rate_ratio (current / reference) is reported as well.
    """
    ref = _group_stats(pd.Series(ref_groups), pd.Series(ref_flags))
    cur = _group_stats(pd.Series(cur_groups), pd.Series(cur_flags))
    rows = {}
    for group in ref.index.intersection(cur.index):
        n_ref, n_cur = int(ref.loc[group, "n"]), int(cur.loc[group, "n"])
        row = {"n_ref": n_ref, "n_cur": n_cur, "share_ref": n_ref / ref["n"].sum(), "share_cur": n_cur / cur["n"].sum(),
               "rate_ref": ref.loc[group, "flagged"] / n_ref, "rate_cur": cur.loc[group, "flagged"] / n_cur}
        if min(n_ref, n_cur) >= min_count:
            f_ref, f_cur = int(ref.loc[group, "flagged"]), int(cur.loc[group, "flagged"])
            own = rate_difference(f_ref, n_ref, f_cur, n_cur, design_effect)
            rest = rate_difference(int(ref["flagged"].sum()) - f_ref, int(ref["n"].sum()) - n_ref,
                                   int(cur["flagged"].sum()) - f_cur, int(cur["n"].sum()) - n_cur, design_effect)
            se_vs_rest = np.sqrt(own["se"] ** 2 + rest["se"] ** 2)  # group and rest are different rows
            row.update(own)
            row["rate_ratio"] = row["rate_cur"] / row["rate_ref"] if row["rate_ref"] > 0 else np.nan
            row.update(diff_vs_rest=own["diff"] - rest["diff"],
                       p_vs_rest=float(2 * stats.norm.sf(abs(own["diff"] - rest["diff"]) / se_vs_rest)))
        rows[group] = row
    table = pd.DataFrame(rows).T.astype(float)
    for column in ("p_value", "p_vs_rest", "diff_vs_rest", "rate_ratio"):
        if column not in table:
            table[column] = np.nan
    tested = table["p_value"].notna()
    cutoff = alpha / max(int(tested.sum()), 1)
    table["significant"] = tested & (table["p_value"] < cutoff)
    table["differs_from_rest"] = tested & (table["p_vs_rest"] < cutoff)
    return table


def swap_effects(
    flag_fn: Callable[[pd.DataFrame], np.ndarray],
    current: pd.DataFrame,
    reference: pd.DataFrame,
    groups: dict[str, list[str]],
    n_repeats: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Diagnostic: how much the flag rate of `current` changes if some columns take their reference values.

    For each group of columns, the columns are replaced once with values drawn at random from `reference`
    and once with their own values in shuffled order (control: both break the link to the other columns, so
    only the distribution differs). The result is flag rate with current values minus flag rate with
    reference values: positive = the current values of this group go with a higher flag rate.

    The columns of a group are drawn together, from the same rows of `reference` (and shuffled together in the
    control), so the relationships inside the group are kept; only the link to the columns outside it is broken.

    The draw is repeated n_repeats times with the seeds seed, seed + 1, ..., so the result is reproducible.
    Returns the mean and the standard deviation over the repetitions for each group. One group is changed at a
    time and the other columns are left as they are, so the effects do not add up to the total change, and they
    say nothing about causes.
    """
    effects = {name: [] for name in groups}
    for repeat in range(n_repeats):
        for name, columns in groups.items():
            rows = reference[columns].sample(len(current), replace=True, random_state=seed + repeat)
            own = current[columns].sample(frac=1, random_state=seed + repeat)
            from_reference, own_shuffled = current.copy(), current.copy()
            from_reference[columns] = rows.to_numpy()
            own_shuffled[columns] = own.to_numpy()
            effects[name].append(float(np.mean(flag_fn(own_shuffled)) - np.mean(flag_fn(from_reference))))
    return pd.DataFrame({name: {"mean": np.mean(v), "std": np.std(v, ddof=1) if len(v) > 1 else np.nan}
                         for name, v in effects.items()}).T
