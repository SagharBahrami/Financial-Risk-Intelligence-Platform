"""Monitor the frozen fraud model month by month, with the labels that are available in BAF.

The model, its threshold and the registry are only read, never changed. The months you compare are used for
reporting only, never to choose anything (see "Evaluation protocol" in the README).

The report keeps three kinds of change apart:
1. Performance: what the decisions do (FPR, recall, precision, fraud rate). Needs labels.
2. Score drift: the distribution of the model scores and the share above the frozen threshold. It is reported twice:
   - all applications: needs NO labels, so this is the version that can be monitored in production;
   - legitimate applications only: needs the labels. The share above the threshold is then the FPR, so this is the
     link between 1 and 3 (FPR can only change through the legitimate scores or through the labels).
3. Input drift: the distribution of the features of all applications (needs no labels).
Subgroups (FPR per group) and the optional feature-swap diagnostic (--swap) use the legitimate applications,
so they need labels. The swap diagnostic shows which features go with the change of the FPR; it is an
association on modified data, not proof of what caused the change.

BAF has no dates or customer ids, so by default every interval treats applications as independent (design
effect 1). That is too optimistic; the report shows how much extra variance each FPR difference can survive.
--design-effect inflates the variance of all probability statements: the monthly Wilson intervals, the FPR
differences and the subgroup tests. It does not change PSI or KS, which are not intervals.

Run with:  python -m risk_platform.fraud.monitor --reference-month 5 --months 6 7
"""
import argparse
import itertools
import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from risk_platform.common.config import PROJECT_ROOT, load_config, resolve_path
from risk_platform.common.drift import (
    compare_subgroups,
    decompose_rate_change,
    feature_drift,
    ks_statistic,
    psi,
    rate_difference,
    split_half_noise,
    swap_effects,
    wilson_interval,
)
from risk_platform.common.metrics import recall_at_fpr
from risk_platform.common.registry import CHAMPION
from risk_platform.common.tracking import setup_mlflow
from risk_platform.fraud.train import split_xy

logger = logging.getLogger(__name__)

SCORE_QUANTILES = [0.5, 0.9, 0.95, 0.99]
FEATURE_GROUPS = {"velocity (6h, 24h, 4w)": ["velocity_6h", "velocity_24h", "velocity_4w"]}


def feature_columns(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    return [c for c in df.columns if c not in (config["target"], config["time_column"], "score", "flag")]


def performance_table(
    scored: pd.DataFrame, threshold: float, months: list[int], config: dict[str, Any], design_effect: float = 1.0
) -> pd.DataFrame:
    """Decisions at the frozen threshold, per month, with 95% Wilson intervals for FPR and recall.

    The intervals use the same design_effect as the FPR differences (1 = independent rows). Needs labels.
    """
    target, time_column = config["target"], config["time_column"]
    rows = {}
    for month in months:
        part = scored[scored[time_column] == month]
        y, flagged = part[target].to_numpy(), part["score"].to_numpy() >= threshold
        n_legit, n_fraud = int((y == 0).sum()), int((y == 1).sum())
        false_pos, true_pos = int((flagged & (y == 0)).sum()), int((flagged & (y == 1)).sum())
        fpr_low, fpr_high = wilson_interval(false_pos, n_legit, design_effect=design_effect)
        recall_low, recall_high = wilson_interval(true_pos, n_fraud, design_effect=design_effect)
        both_classes = n_legit > 0 and n_fraud > 0
        rows[month] = {
            "n": len(part), "n_fraud": n_fraud, "n_legit": n_legit, "fraud_rate": n_fraud / len(part),
            "flag_rate": flagged.mean(), "false_pos": false_pos, "true_pos": true_pos,
            "fpr": false_pos / n_legit, "fpr_low": fpr_low, "fpr_high": fpr_high,
            "recall": true_pos / n_fraud, "recall_low": recall_low, "recall_high": recall_high,
            "precision": true_pos / max(flagged.sum(), 1),
            # ranking quality, independent of the threshold
            "roc_auc": roc_auc_score(y, part["score"]) if both_classes else np.nan,
            "recall_at_max_fpr": recall_at_fpr(y, part["score"], config["evaluation"]["max_fpr"])[0] if both_classes else np.nan,
        }
    return pd.DataFrame(rows).T.astype({"n": int, "n_fraud": int, "n_legit": int, "false_pos": int, "true_pos": int})


def fpr_differences(performance: pd.DataFrame, design_effect: float = 1.0) -> pd.DataFrame:
    """FPR difference for every pair of months (later - earlier), with the variance inflation it can survive."""
    rows = {}
    for earlier, later in itertools.combinations(sorted(performance.index), 2):
        a, b = performance.loc[earlier], performance.loc[later]
        rows[f"{later} - {earlier}"] = rate_difference(
            int(a["false_pos"]), int(a["n_legit"]), int(b["false_pos"]), int(b["n_legit"]), design_effect
        )
    return pd.DataFrame(rows).T


def score_drift_table(
    scored: pd.DataFrame, threshold: float, reference: int, months: list[int], config: dict[str, Any],
    noise_repeats: int = 100, legit_only: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Scores versus the reference month, and the level that sampling noise reaches.

    legit_only=False: all applications. Needs no labels (the target column is not used), so it can be computed in
    production. share_above_threshold is then the flag rate.
    legit_only=True: legitimate applications only. Needs the labels. share_above_threshold is then the FPR, and the
    median score of the fraud cases is added.
    """
    time_column = config["time_column"]
    population = scored[scored[config["target"]] == 0] if legit_only else scored
    by_month = {m: population.loc[population[time_column] == m, "score"].to_numpy() for m in [reference, *months]}
    rows = {}
    for month, values in by_month.items():
        row = {f"q{round(q * 100)}": np.quantile(values, q) for q in SCORE_QUANTILES}
        row["share_above_threshold"] = (values >= threshold).mean()
        # percentile of these scores at which the frozen threshold sits (95 = 5% above it)
        row["threshold_percentile"] = 100 * (values < threshold).mean()
        if legit_only:
            fraud = scored[(scored[config["target"]] == 1) & (scored[time_column] == month)]
            row["fraud_median"] = fraud["score"].median()
        row["ks"] = ks_statistic(by_month[reference], values) if month != reference else np.nan
        row["psi"] = psi(by_month[reference], values) if month != reference else np.nan
        rows[month] = row
    noise = {
        "ks": split_half_noise(by_month[reference], ks_statistic, noise_repeats),
        "psi": split_half_noise(by_month[reference], psi, noise_repeats),
    }
    return pd.DataFrame(rows).T, noise


def subgroup_columns(scored: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    """Categorical columns and two-valued numeric columns (yes/no features)."""
    columns = list(config["categorical_columns"])
    for column in feature_columns(scored, config):
        if column not in columns and scored[column].nunique() == 2:
            columns.append(column)
    return columns


def subgroup_tables(
    scored: pd.DataFrame, threshold: float, reference: int, months: list[int], config: dict[str, Any],
    design_effect: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """False-positive rate per subgroup versus the reference month, and the composition/within split.

    Only legitimate applications are used, so the flag rate is the FPR.
    """
    legit = scored[scored[config["target"]] == 0].assign(flag=lambda d: (d["score"] >= threshold).astype(int))
    ref = legit[legit[config["time_column"]] == reference]
    subgroups, splits = [], {}
    for month in months:
        cur = legit[legit[config["time_column"]] == month]
        for column in subgroup_columns(scored, config):
            table = compare_subgroups(ref[column], ref["flag"], cur[column], cur["flag"], design_effect)
            table.index = pd.MultiIndex.from_product([[month], [column], table.index], names=["month", "feature", "group"])
            subgroups.append(table)
            splits[(month, column)] = decompose_rate_change(ref[column], ref["flag"], cur[column], cur["flag"])
    return pd.concat(subgroups), pd.DataFrame(splits).T.rename_axis(["month", "feature"])


def input_drift_tables(
    scored: pd.DataFrame, reference: int, months: list[int], config: dict[str, Any]
) -> dict[int, pd.DataFrame]:
    """Feature drift of all applications, per month versus the reference month. Needs no labels."""
    columns = feature_columns(scored, config)
    ref = scored.loc[scored[config["time_column"]] == reference, columns]
    return {
        month: feature_drift(
            ref, scored.loc[scored[config["time_column"]] == month, columns], config["categorical_columns"],
            config["features"]["missing_as_negative"],
        )
        for month in months
    }


def swap_tables(
    scored: pd.DataFrame, flag_fn, reference: int, months: list[int], config: dict[str, Any], n_repeats: int = 5
) -> pd.DataFrame:
    """Feature-swap diagnostic (percentage points of FPR) per month versus the reference month.

    Uses the legitimate applications, so it needs labels. Each value is the mean and standard deviation over
    n_repeats random draws with fixed seeds (0 .. n_repeats - 1), so the table is reproducible.
    """
    legit = scored[scored[config["target"]] == 0]
    columns = feature_columns(legit, config)
    ref = legit.loc[legit[config["time_column"]] == reference, columns].reset_index(drop=True)
    families = {name: cols for name, cols in FEATURE_GROUPS.items() if set(cols) <= set(columns)}
    groups = {**{c: [c] for c in columns}, **families}
    effects = {
        month: 100 * swap_effects(
            flag_fn, legit.loc[legit[config["time_column"]] == month, columns].reset_index(drop=True), ref, groups,
            n_repeats=n_repeats,
        )
        for month in months
    }
    return pd.concat(effects, axis=1)  # columns: (month, "mean" / "std")


def build_report(
    scored: pd.DataFrame, threshold: float, reference: int, months: list[int], config: dict[str, Any],
    design_effect: float = 1.0, flag_fn=None, swap_repeats: int = 5,
) -> dict[str, Any]:
    """All tables of the report. `scored` has the raw features, the target, the month and a "score" column."""
    performance = performance_table(scored, threshold, [reference, *months], config, design_effect)
    score_drift_all, noise_all = score_drift_table(scored, threshold, reference, months, config)
    score_drift_legit, noise_legit = score_drift_table(scored, threshold, reference, months, config, legit_only=True)
    subgroups, splits = subgroup_tables(scored, threshold, reference, months, config, design_effect)
    report = {
        "threshold": threshold, "reference": reference, "months": months, "design_effect": design_effect,
        "performance": performance,
        "fpr_differences": fpr_differences(performance, design_effect),
        "score_drift_all": score_drift_all, "score_noise_all": noise_all,
        "score_drift_legit": score_drift_legit, "score_noise_legit": noise_legit,
        "input_drift": input_drift_tables(scored, reference, months, config),
        "subgroups": subgroups, "subgroup_splits": splits,
    }
    if flag_fn is not None:
        report["swap"] = swap_tables(scored, flag_fn, reference, months, config, swap_repeats)
    return report


def _fmt(df: pd.DataFrame, **kwargs) -> str:
    return "```\n" + df.round(4).to_string(**kwargs) + "\n```\n"


def render_report(report: dict[str, Any], calibration_months: list[int] | None = None) -> str:
    """The report as Markdown text."""
    ref, months, threshold = report["reference"], report["months"], report["threshold"]
    out = [f"# Fraud model monitoring: months {months} against month {ref}\n",
           f"Frozen decision threshold: {threshold:.4f}. Rates are fractions (0.05 = 5%). "
           f"Intervals assume independent rows (design effect {report['design_effect']:g}); BAF has no dates or "
           "customer ids, so the real uncertainty is larger.\n"]
    if calibration_months and ref in calibration_months:
        out.append(f"Month {ref} is the month the threshold was chosen on, so its FPR is about the target by "
                   "construction and is not an independent benchmark.\n")

    perf = report["performance"]
    out += ["## 1. Performance (decisions at the frozen threshold)\n",
            _fmt(perf[["n", "n_fraud", "fraud_rate", "flag_rate", "fpr", "fpr_low", "fpr_high", "recall", "precision",
                       "roc_auc", "recall_at_max_fpr"]]),
            "roc_auc and recall_at_max_fpr (the best threshold of each month) measure ranking only, not the decisions.\n",
            "FPR differences (later - earlier). `design_effect_to_lose_significance` is how many times larger the "
            "variance would have to be, for example because of repeated customers, before the 95% interval "
            "reaches zero.\n",
            _fmt(report["fpr_differences"][["diff", "ci_low", "ci_high", "z", "design_effect_to_lose_significance"]])]

    for key, title, note in [
        ("all", "2a. Score drift of ALL applications (no labels needed)",
         "`share_above_threshold` is the flag rate. This is the version that can be monitored in production."),
        ("legit", "2b. Score drift of LEGITIMATE applications (needs labels)",
         "`share_above_threshold` is the FPR. Only available once the outcomes are known."),
    ]:
        noise = report[f"score_noise_{key}"]
        out += [f"## {title}, versus the reference month\n", note + "\n", _fmt(report[f"score_drift_{key}"]),
                f"Noise level (95th percentile between two random halves of month {ref}): KS {noise['ks']:.4f}, "
                f"PSI {noise['psi']:.5f}. PSI rule of thumb: below 0.1 small, 0.1-0.25 moderate, above 0.25 large.\n"]

    out.append("## 3. Input drift (all applications versus the reference month, no labels needed)\n")
    for month, table in report["input_drift"].items():
        top = table.sort_values("psi", ascending=False).head(10)
        out += [f"Month {month}: ten largest feature PSI values (`psi_noise` = level reached by chance).\n",
                _fmt(top[["kind", "psi", "psi_noise", "ks", "missing_ref", "missing_cur"]])]

    sub = report["subgroups"]
    tested = int(sub["p_value"].notna().sum())
    moved = sub[sub["significant"]]
    different = sub[sub["differs_from_rest"]].assign(abs_diff=lambda d: d["diff_vs_rest"].abs())
    out += [f"## 4. Subgroups (FPR versus month {ref}; legitimate applications, needs labels)\n",
            f"{tested} subgroups were tested (at least 200 legitimate applications in both months, Bonferroni "
            f"correction). The FPR changed in {len(moved)} of them, but when the whole score level moves almost every "
            f"subgroup moves with it. {len(different)} changed differently from the rest of the population "
            "(`diff_vs_rest` = the subgroup's change minus the change of all other rows, in absolute points; groups with a high FPR move more in points for the same relative change, so compare `rate_ratio` with the overall ratio). Ten largest:\n",
            _fmt(different.sort_values("abs_diff", ascending=False).head(10)[
                ["share_ref", "share_cur", "rate_ref", "rate_cur", "rate_ratio", "diff", "diff_vs_rest"]]
                if len(different) else pd.DataFrame({"none": []})),
            "Change of the FPR split into composition (groups became more or less common) and within-group "
            "(the same groups are flagged more or less often), per feature, in percentage points. Large "
            "within-group parts in every feature mean the score level moved, not the mix of applicants.\n",
            _fmt(report["subgroup_splits"].mul(100).round(3))]

    if "swap" in report:
        swap = report["swap"]
        strongest = swap.xs("mean", axis=1, level=1).abs().max(axis=1).sort_values(ascending=False).index
        out += ["## 5. Feature-swap diagnostic (percentage points of FPR; legitimate applications, needs labels)\n",
                "How much the FPR changes if one feature takes the reference month's values instead of its own; "
                "positive = the month's values of that feature go with a higher FPR. Mean and standard deviation "
                "over the random draws (fixed seeds, reproducible). Features of a group are drawn together, from the "
                "same rows. One group at a time, the others unchanged, so the effects do not add up to the total "
                "change. **This is an association on modified data, not evidence of what caused the change.**\n",
                _fmt(swap.reindex(strongest).head(12))]
    return "\n".join(out)


def load_months(config: dict[str, Any], months: list[int]) -> pd.DataFrame:
    frames = [pd.read_parquet(resolve_path(config["data"][f"{name}_path"])) for name in ("train", "test")]
    df = pd.concat(frames, ignore_index=True)
    return df[df[config["time_column"]].isin(months)].reset_index(drop=True)


def run(
    config: dict[str, Any], reference: int, months: list[int], design_effect: float, swap: bool, out_dir: Path,
    swap_repeats: int = 5,
) -> dict:
    setup_mlflow(config["mlflow"]["experiment"])
    model = mlflow.pyfunc.load_model(f"models:/{config['registry']['model_name']}@{CHAMPION}")
    threshold = model.metadata.metadata["decision_threshold"]  # the frozen threshold, never recomputed here

    df = load_months(config, [reference, *months])
    df = df.astype({col: str for col in config["categorical_columns"]})  # same input format as serving
    X, _ = split_xy(df, config)
    scored = df.assign(score=model.predict(X)[:, 1])

    flag_fn = (lambda Z: model.predict(Z)[:, 1] >= threshold) if swap else None
    report = build_report(scored, threshold, reference, months, config, design_effect, flag_fn, swap_repeats)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "monitoring_report.md").write_text(
        render_report(report, config["training"]["calibration_months"]), encoding="utf-8"
    )
    report["performance"].to_csv(out_dir / "performance.csv")
    report["fpr_differences"].to_csv(out_dir / "fpr_differences.csv")
    report["score_drift_all"].to_csv(out_dir / "score_drift_all_applications.csv")
    report["score_drift_legit"].to_csv(out_dir / "score_drift_legitimate_only.csv")
    if "swap" in report:
        report["swap"].to_csv(out_dir / "swap_diagnostic.csv")
    report["subgroups"].to_csv(out_dir / "subgroups.csv")
    for month, table in report["input_drift"].items():
        table.to_csv(out_dir / f"input_drift_month{month}.csv")
    logger.info("Report written to %s", out_dir)
    return report


def main() -> None:
    config = load_config("fraud")
    parser = argparse.ArgumentParser(description="Monitor the frozen fraud model month by month")
    parser.add_argument("--reference-month", type=int, default=config["training"]["calibration_months"][0])
    parser.add_argument("--months", type=int, nargs="+", default=config["split"]["test_months"])
    parser.add_argument("--design-effect", type=float, default=1.0,
                        help="variance inflation (1 = independent rows) for the monthly Wilson intervals, the FPR "
                             "difference intervals and the subgroup tests; PSI and KS are not affected")
    parser.add_argument("--swap", action="store_true", help="also run the feature-swap diagnostic (slower)")
    parser.add_argument("--swap-repeats", type=int, default=5, help="random repetitions of the swap diagnostic")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "monitoring" / "fraud")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report = run(config, args.reference_month, args.months, args.design_effect, args.swap, args.out_dir,
                 args.swap_repeats)
    print(render_report(report, config["training"]["calibration_months"]))


if __name__ == "__main__":
    main()
