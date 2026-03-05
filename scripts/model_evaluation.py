"""Generate split and model diagnostic plots from MLflow artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from mgnify_methods.utils.logging import get_logger

from emobon_models.modeling_evaluation import (
    default_plot_output_dir,
    evaluate_train_test_splits,
    load_run_artifacts,
    plot_feature_importances,
    plot_fold_sizes,
    plot_metric_by_group,
    plot_residual_distribution,
    plot_true_vs_pred,
    select_run_dir,
    to_long_predictions,
    top_taxa_for_plotting,
    write_summary_report,
)


logger = get_logger(__name__, level="INFO")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model evaluation plotting."""
    parser = argparse.ArgumentParser(
        description="Generate diagnostics from MLflow run artifacts."
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
            "Defaults to outputs/analysis_<timestamp>/plots."
        ),
    )
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=20,
        help="Number of top features to include in importance plot.",
    )
    parser.add_argument(
        "--top-n-taxa",
        type=int,
        default=5,
        help="Number of high-variance taxa used for focused scatter plot.",
    )
    return parser.parse_args()


def main() -> None:
    """Run artifact-based split evaluation and plot generation."""
    logger.info("Starting model evaluation workflow")
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    mlruns_dir = args.mlruns_dir or repo_root / "mlruns"
    run_dir = select_run_dir(
        mlruns_dir=mlruns_dir,
        run_id=args.run_id,
        experiment_id=args.experiment_id,
    )
    logger.info("Evaluating MLflow run at %s", run_dir)

    artifacts = load_run_artifacts(run_dir)
    split_report = evaluate_train_test_splits(artifacts.fold_metrics)
    long_predictions = to_long_predictions(artifacts.predictions)
    top_taxa = top_taxa_for_plotting(long_predictions, top_n=args.top_n_taxa)

    output_dir = args.output_dir or default_plot_output_dir(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing evaluation artifacts to %s", output_dir)

    split_report["fold_table"].to_csv(
        output_dir / "split_table.csv",
        index=False,
    )

    plot_fold_sizes(artifacts.fold_metrics, output_dir / "split_sizes.png")
    plot_metric_by_group(
        artifacts.fold_metrics,
        metric="mae",
        output_path=output_dir / "mae_by_group.png",
    )
    plot_metric_by_group(
        artifacts.fold_metrics,
        metric="rmse",
        output_path=output_dir / "rmse_by_group.png",
    )

    plot_true_vs_pred(
        long_predictions,
        output_dir / "true_vs_pred_overall.png",
    )
    if top_taxa:
        plot_true_vs_pred(
            long_predictions,
            output_path=output_dir / "true_vs_pred_top_taxa.png",
            taxa=top_taxa,
        )

    plot_residual_distribution(
        long_predictions,
        output_path=output_dir / "residual_distribution.png",
    )
    plot_feature_importances(
        artifacts.feature_importances,
        output_path=output_dir / "feature_importances_topk.png",
        top_k=args.top_k_features,
    )

    summary_path = write_summary_report(
        output_dir=output_dir,
        run_artifacts=artifacts,
        split_report=split_report,
        long_predictions=long_predictions,
    )
    logger.info("Evaluation workflow complete")

    print(f"Evaluation run ID: {artifacts.run_id}")
    print(f"Evaluation outputs saved to: {output_dir}")
    print(f"Summary report: {summary_path}")


if __name__ == "__main__":
    main()
