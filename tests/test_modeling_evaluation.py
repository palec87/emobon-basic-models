"""Tests for MLflow artifact evaluation helpers."""

from pathlib import Path

import pandas as pd

from emobon_models.modeling_evaluation import evaluate_train_test_splits
from emobon_models.modeling_evaluation import load_run_artifacts
from emobon_models.modeling_evaluation import select_run_dir
from emobon_models.modeling_evaluation import to_long_predictions


def _create_fake_run(run_dir: Path) -> None:
    """Create minimal run structure needed for directory selection tests."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.yaml").write_text("status: ok\n", encoding="utf-8")


def test_select_run_dir_picks_requested_run(tmp_path: Path) -> None:
    """Select explicit run ID from a specific experiment directory."""
    mlruns_dir = tmp_path / "mlruns"
    run_a = mlruns_dir / "123" / "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    run_b = mlruns_dir / "123" / "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    _create_fake_run(run_a)
    _create_fake_run(run_b)

    selected = select_run_dir(
        mlruns_dir=mlruns_dir,
        run_id="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        experiment_id="123",
    )
    assert selected.name == "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def test_evaluate_train_test_splits_reports_integrity() -> None:
    """Compute fold-level integrity checks and derived test ratio."""
    fold_metrics = pd.DataFrame(
        {
            "fold": [0, 1, 2],
            "group": ["A", "B", "C"],
            "n_train": [90, 90, 90],
            "n_test": [10, 10, 10],
            "mae": [0.1, 0.2, 0.3],
            "rmse": [0.2, 0.3, 0.4],
        }
    )

    report = evaluate_train_test_splits(fold_metrics)

    assert report["n_folds"] == 3
    assert report["duplicate_groups"] == []
    assert report["invalid_train_folds"] == []
    assert report["invalid_test_folds"] == []
    assert report["consistent_total_samples"] is True
    assert report["total_samples"] == 100
    assert "test_ratio" in report["fold_table"].columns


def test_to_long_predictions_creates_residual_columns() -> None:
    """Create long true/pred rows and residual statistics per taxon."""
    predictions = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "fold": [0, 0],
            "group": ["A", "A"],
            "true__taxon_1": [0.5, 0.6],
            "pred__taxon_1": [0.4, 0.7],
            "true__taxon_2": [0.2, 0.1],
            "pred__taxon_2": [0.3, 0.2],
        }
    )

    long_df = to_long_predictions(predictions)

    assert long_df.shape[0] == 4
    assert {"y_true", "y_pred", "residual", "abs_error", "taxon"}.issubset(
        long_df.columns
    )
    assert set(long_df["taxon"].unique()) == {"taxon_1", "taxon_2"}


def test_load_run_artifacts_normalizes_index_column(tmp_path: Path) -> None:
    """Normalize legacy prediction artifact with an index column name."""
    run_id = "a" * 32
    run_dir = tmp_path / "mlruns" / "123" / run_id
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.yaml").write_text("status: ok\n", encoding="utf-8")

    pd.DataFrame(
        {
            "fold": [0],
            "group": ["A"],
            "n_train": [9],
            "n_test": [1],
            "mae": [0.1],
            "rmse": [0.2],
        }
    ).to_csv(artifacts_dir / "fold_metrics.csv", index=False)

    pd.DataFrame(
        {
            "index": ["s1"],
            "fold": [0],
            "group": ["A"],
            "true__taxon_1": [0.2],
            "pred__taxon_1": [0.3],
        }
    ).to_csv(artifacts_dir / "fold_predictions.csv", index=False)

    pd.DataFrame(
        {
            "feature": ["f1"],
            "importance": [0.5],
        }
    ).to_csv(artifacts_dir / "feature_importances.csv", index=False)

    artifacts = load_run_artifacts(run_dir)
    assert "sample_id" in artifacts.predictions.columns
