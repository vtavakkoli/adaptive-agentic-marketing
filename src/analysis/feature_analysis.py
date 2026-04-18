from __future__ import annotations

import argparse
import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import f_classif

from src.data.prepare import prepare_dataset
from src.features.label_engineering import ACTION_ORDER

FORBIDDEN_BASE = {
    "action_class",
    "label_type",
    "customer_id",
    "need_score",
    "fatigue_score",
    "intrusiveness_risk",
    "offer_relevance",
    "no_action_preferred",
}
LEAKY_KEYWORDS = ("label", "target", "action", "outcome", "post", "decision")


@dataclass
class AnalysisTarget:
    name: str
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series


def _ensure_input_data(processed_dir: Path, raw_dir: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    if not train_path.exists() or not val_path.exists():
        paths = prepare_dataset(dataset=dataset, raw_dir=raw_dir, processed_dir=processed_dir)
    else:
        paths = {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(processed_dir / "test.csv"),
            "all": str(processed_dir / "all.csv"),
            "metadata": str(processed_dir / "metadata.json"),
        }
    return pd.read_csv(paths["train"]), pd.read_csv(paths["val"]), paths


def _excluded_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for col in df.columns:
        reason = None
        if col in FORBIDDEN_BASE:
            reason = "explicit_forbidden"
        elif any(k in col.lower() for k in LEAKY_KEYWORDS):
            reason = "suspected_leakage_keyword"
        elif col.endswith("_class") or col.endswith("_label"):
            reason = "target_like_column"
        if reason:
            rows.append({"column": col, "reason": reason})
    return pd.DataFrame(rows).drop_duplicates()


def _feature_family(col: str) -> str:
    lowered = col.lower()
    if "recency" in lowered or "day" in lowered or "time" in lowered:
        return "recency_timing"
    if any(k in lowered for k in ["touch", "fatigue", "pressure"]):
        return "contact_pressure_fatigue"
    if any(k in lowered for k in ["offer", "campaign", "coupon"]):
        return "offer_campaign"
    if any(k in lowered for k in ["frequency", "basket", "response", "history"]):
        return "customer_behavior_responsiveness"
    if "channel" in lowered:
        return "channel"
    if any(k in lowered for k in ["interaction", "ratio", "cross"]):
        return "engineered_interaction"
    return "other"


def _encode_features(train_x: pd.DataFrame, val_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_x.copy()
    val = val_x.copy()
    for col in train.columns:
        if pd.api.types.is_numeric_dtype(train[col]):
            median = train[col].median()
            train[col] = train[col].fillna(median)
            val[col] = val[col].fillna(median)
        else:
            fill = "__missing__"
            train[col] = train[col].fillna(fill).astype("string")
            val[col] = val[col].fillna(fill).astype("string")
            categories = pd.Index(train[col].unique())
            mapping = {v: i for i, v in enumerate(categories)}
            train[col] = train[col].map(mapping).fillna(-1).astype(float)
            val[col] = val[col].map(mapping).fillna(-1).astype(float)
    return train, val


def _rf_rankings(train_x: pd.DataFrame, val_x: pd.DataFrame, y_train: pd.Series, y_val: pd.Series, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, float | int | str]] = []
    models = []
    for seed in seeds:
        rf = RandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
            min_samples_leaf=2,
        )
        rf.fit(train_x, y_train)
        models.append((seed, rf))
        for col, imp in zip(train_x.columns, rf.feature_importances_, strict=True):
            records.append({"feature": col, "seed": seed, "importance": float(imp)})

    imp_df = pd.DataFrame(records)
    summary = (
        imp_df.groupby("feature", as_index=False)
        .agg(importance_mean=("importance", "mean"), importance_std=("importance", "std"))
        .sort_values("importance_mean", ascending=False)
    )
    summary["stability_cv"] = summary["importance_std"] / summary["importance_mean"].replace(0, np.nan)
    summary["family"] = summary["feature"].map(_feature_family)
    summary["rank"] = np.arange(1, len(summary) + 1)
    summary["cumulative_importance"] = summary["importance_mean"].cumsum()

    best_seed, best_model = max(models, key=lambda pair: pair[1].score(val_x, y_val))
    perm = permutation_importance(
        estimator=best_model,
        X=val_x,
        y=y_val,
        scoring="f1_weighted",
        n_repeats=12,
        random_state=best_seed,
        n_jobs=-1,
    )
    perm_df = pd.DataFrame(
        {
            "feature": train_x.columns,
            "permutation_mean": perm.importances_mean,
            "permutation_std": perm.importances_std,
            "family": [_feature_family(c) for c in train_x.columns],
        }
    ).sort_values("permutation_mean", ascending=False)
    return summary, perm_df, imp_df


def _feature_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    dup_cols: set[str] = set()
    for i, col_i in enumerate(features):
        for col_j in features[i + 1 :]:
            if train_df[col_i].equals(train_df[col_j]):
                dup_cols.add(col_j)
    for col in features:
        s = train_df[col]
        row = {
            "feature": col,
            "dtype": str(s.dtype),
            "missing_rate": float(s.isna().mean()),
            "cardinality": int(s.nunique(dropna=True)),
            "near_constant": bool(s.nunique(dropna=False) <= 1 or s.value_counts(normalize=True, dropna=False).iloc[0] > 0.98),
            "is_duplicate_of_other": col in dup_cols,
            "family": _feature_family(col),
            "train_val_drift": _drift_stat(train_df[col], val_df[col]),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["near_constant", "missing_rate"], ascending=[False, False])


def _drift_stat(train_s: pd.Series, val_s: pd.Series) -> float:
    if pd.api.types.is_numeric_dtype(train_s):
        t = train_s.dropna().astype(float)
        v = val_s.dropna().astype(float)
        if t.empty or v.empty:
            return 0.0
        qs = np.quantile(t, np.linspace(0, 1, 11))
        qs = np.unique(qs)
        if len(qs) < 3:
            return 0.0
        t_bins = pd.cut(t, bins=qs, include_lowest=True)
        v_bins = pd.cut(v, bins=qs, include_lowest=True)
        t_dist = t_bins.value_counts(normalize=True)
        v_dist = v_bins.value_counts(normalize=True)
        idx = t_dist.index.union(v_dist.index)
        t_prob = t_dist.reindex(idx, fill_value=1e-6)
        v_prob = v_dist.reindex(idx, fill_value=1e-6)
        psi = ((t_prob - v_prob) * np.log((t_prob + 1e-6) / (v_prob + 1e-6))).sum()
        return float(psi)
    t_dist = train_s.astype("string").fillna("__na__").value_counts(normalize=True)
    v_dist = val_s.astype("string").fillna("__na__").value_counts(normalize=True)
    idx = t_dist.index.union(v_dist.index)
    return float(np.abs(t_dist.reindex(idx, fill_value=0) - v_dist.reindex(idx, fill_value=0)).sum() / 2)


def _mutual_info(train_x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    disc = [not pd.api.types.is_numeric_dtype(train_x[c]) for c in train_x.columns]
    encoded, _ = _encode_features(train_x, train_x)
    mi = mutual_info_classif(encoded, y, random_state=42, discrete_features=disc)
    out = pd.DataFrame({"feature": train_x.columns, "mutual_information": mi}).sort_values("mutual_information", ascending=False)
    out["family"] = out["feature"].map(_feature_family)
    return out


def _univariate_scores(train_x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    encoded, _ = _encode_features(train_x, train_x)
    f_vals, p_vals = f_classif(encoded, y)
    out = pd.DataFrame({"feature": train_x.columns, "f_score": f_vals, "p_value": p_vals})
    return out.sort_values("f_score", ascending=False)


def _build_targets(train_df: pd.DataFrame, val_df: pd.DataFrame) -> list[AnalysisTarget]:
    targets = [
        AnalysisTarget(
            name="multiclass",
            train_df=train_df,
            val_df=val_df,
            y_train=pd.Categorical(train_df["action_class"], categories=ACTION_ORDER),
            y_val=pd.Categorical(val_df["action_class"], categories=ACTION_ORDER),
        ),
        AnalysisTarget(
            name="stageA",
            train_df=train_df,
            val_df=val_df,
            y_train=(train_df["action_class"] != "do_nothing").astype(int),
            y_val=(val_df["action_class"] != "do_nothing").astype(int),
        ),
    ]
    train_b = train_df[train_df["action_class"] != "do_nothing"].copy()
    val_b = val_df[val_df["action_class"] != "do_nothing"].copy()
    if len(train_b) > 20 and len(val_b) > 20:
        targets.append(
            AnalysisTarget(
                name="stageB",
                train_df=train_b,
                val_df=val_b,
                y_train=pd.Categorical(train_b["action_class"], categories=ACTION_ORDER[1:]),
                y_val=pd.Categorical(val_b["action_class"], categories=ACTION_ORDER[1:]),
            )
        )
    return targets


def _to_img(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _plot_top(summary: pd.DataFrame, perm: pd.DataFrame, title: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    top_rf = summary.head(12).iloc[::-1]
    top_pm = perm.head(12).iloc[::-1]
    axes[0].barh(top_rf["feature"], top_rf["importance_mean"], color="#5271ff")
    axes[0].set_title(f"RF impurity ({title})")
    axes[1].barh(top_pm["feature"], top_pm["permutation_mean"], color="#03a678")
    axes[1].set_title(f"Permutation ({title})")
    return _to_img(fig)


def _plot_cumulative(summary: pd.DataFrame, title: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(summary["rank"], summary["cumulative_importance"], marker="o", linewidth=1.5)
    ax.set_title(f"Cumulative importance ({title})")
    ax.set_xlabel("Feature rank")
    ax.set_ylabel("Cumulative importance")
    return _to_img(fig)


def _recommendations(feature_summary: pd.DataFrame, rf_multi: pd.DataFrame, perm_multi: pd.DataFrame, exclusions: pd.DataFrame) -> dict[str, object]:
    merged = rf_multi.merge(perm_multi[["feature", "permutation_mean"]], on="feature", how="left")
    merged = merged.merge(feature_summary[["feature", "missing_rate", "near_constant", "train_val_drift"]], on="feature", how="left")
    keep = merged[(merged["importance_mean"] > merged["importance_mean"].quantile(0.6)) & (merged["permutation_mean"] > 0)].sort_values("importance_mean", ascending=False)
    drop = merged[(merged["near_constant"]) | (merged["missing_rate"] > 0.6) | (merged["permutation_mean"] < 0)].sort_values("permutation_mean")
    investigate = merged[(merged["train_val_drift"] > 0.2) | (merged["stability_cv"] > 0.6)].sort_values("train_val_drift", ascending=False)
    return {
        "recommended_keep": keep["feature"].head(30).tolist(),
        "recommended_drop": sorted(set(drop["feature"].head(30).tolist()) | set(exclusions["column"].tolist())),
        "recommended_investigate": investigate["feature"].head(30).tolist(),
        "engineered_features_to_add": [
            "recency_days_x_prior_response_rate",
            "campaign_touches_30d_x_recency_days",
            "avg_basket_value_x_frequency_7d",
            "channel_x_offer_id",
            "rolling_response_rate_30d",
        ],
        "remove_for_leakage_or_proxy": exclusions["column"].tolist(),
        "subsets": {
            "adaptive_framework": keep["feature"].head(18).tolist(),
            "adaptive_ppo_agent_warm_start": keep["feature"].head(12).tolist(),
            "stageA_optional": keep["feature"].head(10).tolist(),
            "stageB_optional": keep["feature"].head(14).tolist(),
        },
    }


def _render_html(output_html: Path, context: dict[str, object]) -> None:
    def _table(df: pd.DataFrame, n: int = 20) -> str:
        return df.head(n).to_html(index=False, classes="tbl", border=0, float_format=lambda x: f"{x:.4f}")

    sections = []
    plots: dict[str, str] = context["plots"]  # type: ignore[assignment]
    for name, data in context["targets"].items():  # type: ignore[union-attr]
        sections.append(
            f"<h2>Target: {name}</h2>"
            f"<p><b>Top RF vs permutation:</b></p><img src='data:image/png;base64,{plots[name + '_top']}' />"
            f"<p><b>Cumulative importance:</b></p><img src='data:image/png;base64,{plots[name + '_cum']}' />"
            f"<h3>RF Importances</h3>{_table(data['rf'])}"
            f"<h3>Permutation Importances</h3>{_table(data['perm'])}"
            f"<h3>Family-level importance</h3>{_table(data['family'])}"
        )

    html = f"""
    <html><head><meta charset='utf-8'><title>Feature Analysis</title>
    <style>
    body {{ font-family: Arial, sans-serif; margin: 22px; line-height:1.35; }}
    h1,h2,h3 {{ color:#1d3557; }}
    .muted {{ color:#666; }}
    .warn {{ background:#fff4e5; border-left:4px solid #fb8500; padding:10px; margin: 8px 0; }}
    .tbl {{ border-collapse: collapse; font-size: 12px; margin-bottom: 18px; }}
    .tbl th,.tbl td {{ border:1px solid #ddd; padding:6px; }}
    img {{ max-width: 100%; border:1px solid #ddd; margin-bottom: 16px; }}
    </style></head><body>
    <h1>Deep Feature Analysis Report</h1>
    <p class='muted'>Timestamp: {context['timestamp']} | Source train/validation only</p>
    <h2>Dataset summary</h2>
    <p>Train rows: {context['train_rows']} | Validation rows: {context['val_rows']} | Predictor count: {context['predictor_count']}</p>
    <p>Analyzed paths: train={context['paths']['train']} ; val={context['paths']['val']}</p>
    <h2>Target definitions</h2>
    <ul>
      <li>multiclass: action_class ∈ do_nothing / defer_action / send_information / send_reminder</li>
      <li>stageA: is_action = action_class != do_nothing</li>
      <li>stageB: action-only rows with classes defer_action / send_information / send_reminder</li>
    </ul>
    <h2>Excluded columns audit</h2>
    {_table(context['exclusions'], 80)}
    <h2>Feature summary diagnostics</h2>
    {_table(context['feature_summary'], 80)}
    <div class='warn'>Scientific safeguard: pipeline fails if forbidden/leaky columns enter predictor set.</div>
    {''.join(sections)}
    <h2>Method agreement/disagreement</h2>
    {_table(context['agreement'], 40)}
    <h2>Leakage / circularity warnings</h2>
    <ul>{''.join(f'<li>{w}</li>' for w in context['warnings'])}</ul>
    <h2>Feature engineering recommendations</h2>
    <pre>{json.dumps(context['recommendations'], indent=2)}</pre>
    <h2>Next-step actions</h2>
    <ol>
      <li>Adopt recommended_keep for adaptive_framework training config.</li>
      <li>Use adaptive_ppo_agent_warm_start subset for supervised warm-start encoder.</li>
      <li>Add engineered interaction features and rerun this report.</li>
    </ol>
    </body></html>
    """
    output_html.write_text(html, encoding="utf-8")


def run_feature_analysis(processed_dir: Path, raw_dir: Path, dataset: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, paths = _ensure_input_data(processed_dir, raw_dir, dataset)
    exclusions = _excluded_columns(train_df)
    excluded = set(exclusions["column"]) if not exclusions.empty else set()
    predictors = [c for c in train_df.columns if c not in excluded]

    forbidden_in_predictors = sorted(FORBIDDEN_BASE.intersection(predictors))
    if forbidden_in_predictors:
        raise ValueError(f"Forbidden columns leaked into predictors: {forbidden_in_predictors}")

    feature_summary = _feature_summary(train_df, val_df, predictors)
    feature_summary.to_csv(reports_dir / "feature_summary.csv", index=False)
    exclusions.to_csv(reports_dir / "feature_exclusion_audit.csv", index=False)

    targets = _build_targets(train_df, val_df)
    seed_list = [13, 29, 42, 77, 101]
    target_artifacts: dict[str, dict[str, pd.DataFrame]] = {}
    plots: dict[str, str] = {}
    warnings = [
        "Proxy-policy helper score columns were force-excluded from modeling.",
        "Test split was intentionally not used for feature selection decisions.",
    ]

    for target in targets:
        tx_train = target.train_df[predictors]
        tx_val = target.val_df[predictors]
        enc_train, enc_val = _encode_features(tx_train, tx_val)
        rf_sum, perm_sum, _seed_level = _rf_rankings(enc_train, enc_val, target.y_train, target.y_val, seed_list)
        family = rf_sum.groupby("family", as_index=False).agg(importance=("importance_mean", "sum")).sort_values("importance", ascending=False)
        rf_sum.to_csv(reports_dir / f"rf_feature_importance_{target.name}.csv", index=False)
        perm_sum.to_csv(reports_dir / f"rf_permutation_importance_{target.name}.csv", index=False)

        target_artifacts[target.name] = {"rf": rf_sum, "perm": perm_sum, "family": family}
        plots[target.name + "_top"] = _plot_top(rf_sum, perm_sum, target.name)
        plots[target.name + "_cum"] = _plot_cumulative(rf_sum, target.name)

    mi = _mutual_info(train_df[predictors], pd.Categorical(train_df["action_class"]))
    uni = _univariate_scores(train_df[predictors], pd.Categorical(train_df["action_class"]))
    agreement = target_artifacts["multiclass"]["rf"][["feature", "rank"]].merge(
        target_artifacts["multiclass"]["perm"][["feature"]].reset_index().rename(columns={"index": "perm_rank"}), on="feature"
    ).merge(mi[["feature"]].reset_index().rename(columns={"index": "mi_rank"}), on="feature")
    agreement["rank_disagreement"] = (agreement["rank"] - (agreement["perm_rank"] + 1)).abs()
    agreement = agreement.sort_values("rank_disagreement", ascending=False)

    recs = _recommendations(feature_summary, target_artifacts["multiclass"]["rf"], target_artifacts["multiclass"]["perm"], exclusions)
    (reports_dir / "feature_recommendations.json").write_text(json.dumps(recs, indent=2), encoding="utf-8")

    context = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "predictor_count": len(predictors),
        "paths": paths,
        "exclusions": exclusions,
        "feature_summary": feature_summary,
        "targets": target_artifacts,
        "plots": plots,
        "agreement": agreement.merge(mi, on="feature", how="left").merge(uni, on="feature", how="left"),
        "warnings": warnings,
        "recommendations": recs,
    }
    report_path = reports_dir / "feature_analysis.html"
    _render_html(report_path, context)
    # acceptance alias
    (output_dir / "feature.html").write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    print("Feature analysis complete")
    print(f"Analyzed dataset path(s): train={paths['train']} val={paths['val']}")
    print(f"Included predictor count: {len(predictors)}")
    print(f"Excluded predictor count: {len(excluded)}")
    print("Top 20 recommended features:", ", ".join(recs["recommended_keep"][:20]))
    print("Top forbidden/leaky excluded features:", ", ".join(sorted(excluded)[:20]))
    print(f"Output paths: {report_path}, {output_dir / 'feature.html'}")
    print("Next commands:")
    print("- python -m src.pipeline.run_experiment --mode adaptive_framework --config configs/adaptive_hierarchical.yaml")
    print("- python -m src.rl.train_ppo --config configs/adaptive_hierarchical.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deep feature analysis with RF diagnostics")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--raw-dir", default="data/raw/synthetic")
    parser.add_argument("--dataset", choices=["synthetic", "dunnhumby"], default="synthetic")
    parser.add_argument("--output-dir", default="outputs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_feature_analysis(
        processed_dir=Path(args.processed_dir),
        raw_dir=Path(args.raw_dir),
        dataset=args.dataset,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
