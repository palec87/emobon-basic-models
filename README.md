# emobon-models

[![tests](https://img.shields.io/badge/tests-15%20passed-brightgreen)](tests/)
[![python](https://img.shields.io/badge/python-%3E%3D3.12-blue)](pyproject.toml)

EMOBON metadata-to-abundance modeling with group-based LOOCV, MLflow
tracking, optional hyperparameter tuning, and post-run evaluation plots.

## 1. Setup the Repository

### Prerequisites

- Python `>=3.12`
- `uv` installed (recommended workflow)

### Clone and install

```bash
git clone https://github.com/palec87/emobon-models.git
cd emobon-models
uv sync --extra dev --extra docs
```

### Verify installation

```bash
uv run pytest -q
```

If you prefer Make targets:

```bash
make dev-install
make test
```

## 2. Example NB

The essence of the package is demonstrated in the `nbs/teaching_example.ipynb`, where EMO-BON data are loaded, preprocessed and model fitted. The rest of the package lies in the LOOCV strategy, logging and hyperparameter tuning.

## 3. Config and Script Structure (Brief)

### Main config file

- `configs/model_test.json`

Key sections:

- `feature`: grouping column used for outer LOOCV (for example,
	`"observatory ID"`).
- `modeling.model_type`: selects estimator:
	`random_forest`, `ridge`, `pls`, `elasticnet`.
- `modeling.random_forest|ridge|pls|elasticnet`: model-specific
	hyperparameters.
- `modeling.tuning`: optional nested inner-CV grid search:
	- `enabled`
	- `method` (`grid_search`)
	- `inner_cv_folds`
	- `max_candidates`
	- `grids.<model_name>`
- `modeling.mlflow`: tracking URI, experiment name, run name.
- `modeling.metadata_cols`: metadata subset used for training.

### Main scripts

- `scripts/model_testing.py`:
	runs preprocessing (or cache load), then model training with group LOOCV,
	and logs artifacts/metrics to MLflow.
- `scripts/model_evaluation.py`:
	loads MLflow artifacts and generates diagnostics plots plus summary report.

Typical workflow:

```bash
uv run python scripts/model_testing.py
uv run python scripts/model_evaluation.py --experiment-id <EXPERIMENT_ID>
```

## 4. Where to Find Evaluation Scores

### MLflow run outputs (`mlruns/`)

Per run directory:

- `mlruns/<experiment_id>/<run_id>/metrics/fold_mae`
- `mlruns/<experiment_id>/<run_id>/metrics/fold_rmse`
- `mlruns/<experiment_id>/<run_id>/metrics/mae_mean`
- `mlruns/<experiment_id>/<run_id>/metrics/rmse_mean`

Detailed artifacts:

- `mlruns/<experiment_id>/<run_id>/artifacts/fold_metrics.csv`
	- fold-level `mae`, `rmse`, `n_train`, `n_test`, group name
- `mlruns/<experiment_id>/<run_id>/artifacts/fold_predictions.csv`
	- `true__*` and `pred__*` values for sample-level inspection
- `mlruns/<experiment_id>/<run_id>/artifacts/feature_importances.csv`
	- feature statistics (importance or coefficient-based score)

When tuning is enabled, additional artifacts are written:

- `mlruns/<experiment_id>/<run_id>/artifacts/fold_best_params.csv`
- `mlruns/<experiment_id>/<run_id>/artifacts/fold_inner_cv_scores.csv`

### Plot-based evaluation outputs (`outputs/`)

- `outputs/analysis_<timestamp>/plots/summary.md`
- `outputs/analysis_<timestamp>/plots/split_table.csv`
- `outputs/analysis_<timestamp>/plots/*.png`

Notable plot files:

- `mae_by_group.png`
- `rmse_by_group.png`
- `split_sizes.png`
- `true_vs_pred_overall.png`
- `residual_distribution.png`

## Useful Commands

```bash
make test
make lint
make format
make docs
```

## License

See `LICENSE`.

