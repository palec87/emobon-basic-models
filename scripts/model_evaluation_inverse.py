"""Generate diagnostic plots for inverse-model MLflow artifacts.

This script evaluates runs produced by
``run_inverse_group_loocv_with_mlflow`` where taxonomy/abundance
features are used to predict metadata targets that may include both
numeric and categorical columns.

It is intentionally robust to mixed target outputs:
- Numeric targets: true-vs-pred scatter and residual distribution.
- Categorical targets: per-target accuracy bar chart.
- Fold metrics: plots every available metric column by group.
- Feature importances: supports combined table and target-type subsets.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mgnify_methods.utils.logging import get_logger

from emobon_models.modeling_evaluation import (
    evaluate_train_test_splits,
    select_run_dir,
)


logger = get_logger(__name__, level="INFO")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for inverse model evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate diagnostics for inverse-model MLflow artifacts."
        )
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="MLflow run ID. If omitted, latest run is selected.",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Optional experiment ID used to narrow run search.",
    )
    parser.add_argument(
        "--mlruns-dir",
        type=Path,
        default=None,
        help="Path to mlruns root. Defaults to <repo>/mlruns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to save plots and summary. "
            "Defaults to outputs/analysis_inverse_<timestamp>/plots."
        ),
    )
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=20,
        help="Number of top features to include in importance plots.",
    )
    return parser.parse_args()


def _get_pyplot_module() -> Any:
    """Import matplotlib.pyplot lazily with clear failure message."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        msg = "matplotlib is required for evaluation plotting"
        raise ImportError(msg) from exc
    return plt


def _default_output_dir(root_dir: Path) -> Path:
    """Build timestamped inverse-evaluation output directory path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return root_dir / "outputs" / f"analysis_inverse_{timestamp}" / "plots"


def _load_inverse_run_artifacts(
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load inverse-run CSV artifacts from an MLflow run directory.

    Parameters
    ----------
    run_dir:
        Path to a single MLflow run directory containing the ``artifacts``
        subfolder.

    Returns
    -------
    tuple
        ``(fold_metrics, predictions, feature_importances)`` dataframes.
    """
    artifacts_dir = run_dir / "artifacts"
    fold_metrics_path = artifacts_dir / "fold_metrics.csv"
    predictions_path = artifacts_dir / "fold_predictions.csv"
    importances_path = artifacts_dir / "feature_importances.csv"

    for path in [
        fold_metrics_path,
        predictions_path,
        importances_path,
    ]:
        if not path.exists():
            msg = f"Missing artifact: {path}"
            raise FileNotFoundError(msg)

    fold_metrics = pd.read_csv(fold_metrics_path)
    predictions = pd.read_csv(predictions_path)
    feature_importances = pd.read_csv(importances_path)

    required_fold_cols = {"fold", "group", "n_train", "n_test"}
    missing_fold = required_fold_cols.difference(fold_metrics.columns)
    if missing_fold:
        msg = f"fold_metrics.csv missing required columns: {missing_fold}"
        raise ValueError(msg)

    if "sample_id" not in predictions.columns:
        if "index" in predictions.columns:
            predictions = predictions.rename(columns={"index": "sample_id"})
        else:
            msg = (
                "fold_predictions.csv missing sample identifier column; "
                "expected 'sample_id'"
            )
            raise ValueError(msg)

    if "importance" not in feature_importances.columns:
        msg = "feature_importances.csv missing 'importance' column"
        raise ValueError(msg)

    true_cols = [
        c for c in predictions.columns if c.startswith("true__")
    ]
    pred_cols = [
        c for c in predictions.columns if c.startswith("pred__")
    ]
    if not true_cols or not pred_cols:
        msg = (
            "fold_predictions.csv must include true__* and pred__* columns"
        )
        raise ValueError(msg)

    return fold_metrics, predictions, feature_importances


def _paired_target_columns(predictions: pd.DataFrame) -> list[str]:
    """Return target names with both true and predicted columns."""
    true_targets = {
        c.removeprefix("true__")
        for c in predictions.columns
        if c.startswith("true__")
    }
    pred_targets = {
        c.removeprefix("pred__")
        for c in predictions.columns
        if c.startswith("pred__")
    }
    return sorted(true_targets.intersection(pred_targets))


def _build_long_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Convert wide true/pred columns to long format with target labels."""
    rows: list[pd.DataFrame] = []
    for target in _paired_target_columns(predictions):
        true_col = f"true__{target}"
        pred_col = f"pred__{target}"
        long_part = predictions[
            ["sample_id", "fold", "group", true_col, pred_col]
        ].rename(columns={true_col: "y_true", pred_col: "y_pred"})
        long_part["target"] = target
        rows.append(long_part)

    if not rows:
        msg = "No matched true__/pred__ pairs found in predictions"
        raise ValueError(msg)

    long_df = pd.concat(rows, axis=0, ignore_index=True)

    # Numeric casting determines which targets support residual analysis.
    long_df["y_true_num"] = pd.to_numeric(
        long_df["y_true"], errors="coerce"
    )
    long_df["y_pred_num"] = pd.to_numeric(
        long_df["y_pred"], errors="coerce"
    )
    long_df["is_numeric"] = (
        long_df["y_true_num"].notna() & long_df["y_pred_num"].notna()
    )
    long_df["residual"] = long_df["y_true_num"] - long_df["y_pred_num"]
    long_df["abs_error"] = long_df["residual"].abs()
    return long_df


def _target_type_summary(long_df: pd.DataFrame) -> dict[str, list[str]]:
    """Classify each target as numeric or categorical using long data."""
    numeric_targets: list[str] = []
    categorical_targets: list[str] = []

    for target, group_df in long_df.groupby("target", dropna=False):
        if bool(group_df["is_numeric"].all()):
            numeric_targets.append(str(target))
        else:
            categorical_targets.append(str(target))

    return {
        "numeric": sorted(numeric_targets),
        "categorical": sorted(categorical_targets),
    }


def _plot_fold_sizes(
    fold_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot per-fold train/test sample counts."""
    plt = _get_pyplot_module()
    ordered = fold_metrics.sort_values("fold")

    x_labels = [f"{r.fold}: {r.group}" for r in ordered.itertuples()]
    positions = np.arange(len(ordered))
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        positions - width / 2,
        ordered["n_train"],
        width=width,
        label="n_train",
        color="#2b6cb0",
    )
    ax.bar(
        positions + width / 2,
        ordered["n_test"],
        width=width,
        label="n_test",
        color="#dd6b20",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel("Sample count")
    ax.set_title("Inverse Model: Train/Test Sample Counts by Fold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _metric_columns(fold_metrics: pd.DataFrame) -> list[str]:
    """Return metric columns available for fold-level plotting."""
    ignored = {"fold", "group", "n_train", "n_test"}
    metrics = [
        c for c in fold_metrics.columns
        if c not in ignored and pd.api.types.is_numeric_dtype(fold_metrics[c])
    ]
    return metrics


def _plot_metric_by_group(
    fold_metrics: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    """Plot one metric across held-out groups."""
    if metric not in fold_metrics.columns:
        msg = f"Metric column not found: {metric}"
        raise ValueError(msg)

    plt = _get_pyplot_module()
    ordered = fold_metrics.sort_values(metric, ascending=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(
        ordered["group"].astype(str),
        ordered[metric],
        color="#2f855a",
    )
    ax.set_ylabel(metric)
    ax.set_xlabel("Held-out group")
    ax.set_title(f"{metric} by Held-out Group")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_true_vs_pred_numeric(
    long_df: pd.DataFrame,
    output_path: Path,
) -> bool:
    """Plot true-vs-pred scatter for numeric targets.

    Returns ``True`` when a plot is produced and ``False`` when no
    numeric predictions are available.
    """
    plt = _get_pyplot_module()
    numeric_df = long_df[long_df["is_numeric"]]
    if numeric_df.empty:
        return False

    x = numeric_df["y_true_num"].to_numpy()
    y = numeric_df["y_pred_num"].to_numpy()
    lower = float(min(x.min(), y.min()))
    upper = float(max(x.max(), y.max()))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, alpha=0.35, s=10, color="#2b6cb0")
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="#c53030")
    ax.set_xlabel("True value")
    ax.set_ylabel("Predicted value")
    ax.set_title("Inverse Model: True vs Predicted (Numeric Targets)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def _plot_residual_distribution_numeric(
    long_df: pd.DataFrame,
    output_path: Path,
) -> bool:
    """Plot residual histogram for numeric targets.

    Returns ``True`` when a plot is produced and ``False`` when no
    numeric predictions are available.
    """
    plt = _get_pyplot_module()
    numeric_df = long_df[long_df["is_numeric"]]
    if numeric_df.empty:
        return False

    residuals = numeric_df["residual"].dropna().to_numpy()
    if residuals.size == 0:
        return False

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=40, color="#805ad5", alpha=0.85)
    ax.axvline(0.0, color="#1a202c", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (true - predicted)")
    ax.set_ylabel("Count")
    ax.set_title("Inverse Model: Residual Distribution (Numeric Targets)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def _per_target_categorical_accuracy(
    long_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-target categorical accuracy table."""
    cat_df = long_df[~long_df["is_numeric"]].copy()
    if cat_df.empty:
        return pd.DataFrame(columns=["target", "accuracy", "n_samples"])

    rows: list[dict[str, Any]] = []
    for target, target_df in cat_df.groupby("target", dropna=False):
        # Keep rows where ground truth is present.
        valid = target_df[target_df["y_true"].notna()]
        if valid.empty:
            continue
        correct = (
            valid["y_true"].astype(str) == valid["y_pred"].astype(str)
        )
        rows.append(
            {
                "target": str(target),
                "accuracy": float(correct.mean()),
                "n_samples": int(len(valid)),
            }
        )

    return pd.DataFrame(rows).sort_values("accuracy", ascending=False)


def _plot_categorical_accuracy(
    per_target_acc: pd.DataFrame,
    output_path: Path,
) -> bool:
    """Plot per-target categorical accuracy bars.

    Returns ``True`` when a plot is produced and ``False`` otherwise.
    """
    if per_target_acc.empty:
        return False

    plt = _get_pyplot_module()
    ordered = per_target_acc.sort_values("accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(ordered["target"], ordered["accuracy"], color="#2f855a")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Categorical target")
    ax.set_title("Inverse Model: Per-target Categorical Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def _plot_feature_importances(
    feature_importances: pd.DataFrame,
    output_path: Path,
    top_k: int,
    title: str,
) -> bool:
    """Plot top-k feature importances from a feature importance table.

    Returns ``True`` when a plot is produced and ``False`` when the
    input table is empty.
    """
    if top_k <= 0:
        msg = "top_k must be greater than zero"
        raise ValueError(msg)
    if feature_importances.empty:
        return False

    plt = _get_pyplot_module()
    top_df = feature_importances.sort_values(
        "importance", ascending=False
    ).head(top_k)
    top_df = top_df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        top_df["feature"].astype(str),
        top_df["importance"],
        color="#975a16",
    )
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def _write_inverse_summary(
    output_dir: Path,
    run_id: str,
    experiment_id: str,
    fold_metrics: pd.DataFrame,
    split_report: dict[str, Any],
    long_df: pd.DataFrame,
    per_target_acc: pd.DataFrame,
    target_summary: dict[str, list[str]],
) -> Path:
    """Write markdown summary for inverse model evaluation."""
    numeric_df = long_df[long_df["is_numeric"]]
    median_abs_error = (
        float(numeric_df["abs_error"].median())
        if not numeric_df.empty else None
    )

    lines = [
        "# Inverse Modeling Evaluation Summary",
        "",
        f"Run ID: `{run_id}`",
        f"Experiment ID: `{experiment_id}`",
        "",
        "## Split Integrity",
        f"- Number of folds: {split_report['n_folds']}",
        f"- Unique held-out groups: {split_report['unique_groups']}",
        (
            "- Duplicate held-out groups: "
            f"{split_report['duplicate_groups'] or 'None'}"
        ),
        (
            "- Invalid train folds (n_train <= 0): "
            f"{split_report['invalid_train_folds'] or 'None'}"
        ),
        (
            "- Invalid test folds (n_test <= 0): "
            f"{split_report['invalid_test_folds'] or 'None'}"
        ),
        "",
        "## Targets",
        (
            "- Numeric targets "
            f"({len(target_summary['numeric'])}): "
            f"{target_summary['numeric'] or 'None'}"
        ),
        (
            "- Categorical targets "
            f"({len(target_summary['categorical'])}): "
            f"{target_summary['categorical'] or 'None'}"
        ),
        "",
        "## Fold Metrics (Means)",
    ]

    for metric in _metric_columns(fold_metrics):
        lines.append(
            f"- {metric}: {float(fold_metrics[metric].mean()):.6f}"
        )

    if median_abs_error is not None:
        lines.append(
            "- Median absolute residual (numeric targets): "
            f"{median_abs_error:.6f}"
        )

    if not per_target_acc.empty:
        lines.extend([
            "",
            "## Best Categorical Targets by Accuracy",
        ])
        for row in per_target_acc.head(5).itertuples():
            lines.append(
                f"- {row.target}: accuracy={row.accuracy:.4f} "
                f"(n={row.n_samples})"
            )

    summary_path = output_dir / "summary_inverse.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote inverse summary report to %s", summary_path)
    return summary_path


def main() -> None:
    """Run inverse-model artifact evaluation and generate plots."""
    logger.info("Starting inverse model evaluation workflow")
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    mlruns_dir = args.mlruns_dir or repo_root / "mlruns"

    run_dir = select_run_dir(
        mlruns_dir=mlruns_dir,
        run_id=args.run_id,
        experiment_id=args.experiment_id,
    )
    logger.info("Evaluating inverse MLflow run at %s", run_dir)

    fold_metrics, predictions, feature_importances = (
        _load_inverse_run_artifacts(run_dir)
    )
    split_report = evaluate_train_test_splits(fold_metrics)
    long_df = _build_long_predictions(predictions)
    target_summary = _target_type_summary(long_df)

    output_dir = args.output_dir or _default_output_dir(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing inverse evaluation artifacts to %s", output_dir)

    split_report["fold_table"].to_csv(
        output_dir / "split_table.csv", index=False
    )

    _plot_fold_sizes(fold_metrics, output_dir / "split_sizes.png")

    for metric in _metric_columns(fold_metrics):
        _plot_metric_by_group(
            fold_metrics,
            metric=metric,
            output_path=output_dir / f"{metric}_by_group.png",
        )

    wrote_numeric_scatter = _plot_true_vs_pred_numeric(
        long_df,
        output_dir / "true_vs_pred_numeric.png",
    )
    wrote_residuals = _plot_residual_distribution_numeric(
        long_df,
        output_dir / "residual_distribution_numeric.png",
    )

    per_target_acc = _per_target_categorical_accuracy(long_df)
    wrote_cat_acc = _plot_categorical_accuracy(
        per_target_acc,
        output_dir / "categorical_accuracy_by_target.png",
    )

    _plot_feature_importances(
        feature_importances,
        output_dir / "feature_importances_topk_overall.png",
        top_k=args.top_k_features,
        title=(
            f"Top {args.top_k_features} Feature Importances (Overall)"
        ),
    )

    if "target_type" in feature_importances.columns:
        for target_type in ["regression", "classification"]:
            subset = feature_importances[
                feature_importances["target_type"] == target_type
            ]
            _plot_feature_importances(
                subset,
                output_dir /
                f"feature_importances_topk_{target_type}.png",
                top_k=args.top_k_features,
                title=(
                    f"Top {args.top_k_features} Feature Importances "
                    f"({target_type})"
                ),
            )

    summary_path = _write_inverse_summary(
        output_dir=output_dir,
        run_id=run_dir.name,
        experiment_id=run_dir.parent.name,
        fold_metrics=fold_metrics,
        split_report=split_report,
        long_df=long_df,
        per_target_acc=per_target_acc,
        target_summary=target_summary,
    )

    logger.info("Inverse evaluation workflow complete")
    print(f"Inverse evaluation run ID: {run_dir.name}")
    print(f"Evaluation outputs saved to: {output_dir}")
    print(f"Summary report: {summary_path}")
    print(f"Generated numeric scatter: {wrote_numeric_scatter}")
    print(f"Generated numeric residual plot: {wrote_residuals}")
    print(f"Generated categorical accuracy plot: {wrote_cat_acc}")


if __name__ == "__main__":
    main()
