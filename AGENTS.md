# Song Analysis Project Guide

## Overview

Predicts popularity of Trinidad dancehall tracks using a model trained on hip-hop audio features. Entry point: `song-analysis/train_models.py` (run from repo root).

## Entry Point

```bash
python song-analysis/train_models.py
```

Reads data from `song-analysis/data/hip_hop-track-attributes.csv`, trains four models, saves artifacts to `song-analysis/model_artifacts/`.

## Project Structure

```
song-analysis/
├── train_models.py             # orchestration script
├── requirements.txt            # Python dependencies (scikit-learn, pandas, etc.)
├── data/                       # CSV datasets (hip_hop, dancehall hip-hop, trini_dancehall)
├── src/
│   ├── data_processing/        # feature lists, transforms, splits
│   ├── model_training/         # fit_model.py (grid search), evaluate_model.py
│   └── types/                  # TrackAttributeColumns enum
├── model_artifacts/            # trained models (.joblib), train/test splits
└── docs/                       # next_steps.md TODO list
```

## Pipeline

1. Load CSV → 2. Add derived features (release_year, max_aggr_followers) → 3. Split/encode/scale (80/20, seed=200294814) → 4. Grid search models → 5. Save artifacts

### Models (all use 5-fold CV with RMSE scoring):

| Model | Search space |
|-------|--------------|
| ElasticNet | alpha: [0.1,1], l1_ratio: [0.1,1] |
| PoissonRegressor | alpha: [0.1,1] |
| RandomForest | n_estimators: 100-550, max_depth: 4-8, criterion="poisson" |
| HistGradientBoostingRegressor | max_iter: 100-700, max_features: 0.4-1.0, loss="poisson" |

## Key Constants (`src/data_processing/__init__.py`)

- `TARGET_RESPONSE = "popularity"`
- `TARGET_AUDIO_FEATURE_LIST` (14 features)
- `TRACK_METADATA_FEATURE_LIST` (release_year, max_aggr_followers)
- `CATEGORICAL_COLS = ["mode", "key", "time_signature"]`
- `AUDIO_FEATURE_DTYPE_MAP`, `ALL_FEATURE_DTYPE_MAP`

## Important Notes

### Data sources

Datasets from [song-feature-extraction](https://github.com/rparmasar/song-feature-extraction). Pull new data before re-running for fresh results.

### Artifacts

`joblib` files in `model_artifacts/` enable reproducibility. Don't modify after training.

### Dancehall-specific constraint

Dropping `main_artist` feature because it causes errors with dancehall tracks (see `src/data_processing/feature_transforms.py:37-40`).

### Environment

Uses virtual environment in `env/`. Activate before installing dependencies or running scripts.

## Common Commands

```bash
# Install dependencies (in the env)
pip install -r song-analysis/requirements.txt

# Run full pipeline
python song-analysis/train_models.py

# Use loaded models to score new tracks
# via src/model_training/*.py
```

## Verification Checklist

Before considering a change complete:

1. Data: Confirm correct CSV file in `data/` directory
2. Artifacts: Check `model_artifacts/` has expected `.joblib` files
3. Models: Verify `train_times_df.csv` exists with all four models
4. Reproducibility: Ensure seed=200294814 is used in any new splitting code

## Notebook Reading Tips

## Notebook Reading Skill

### How it works
- Reads notebooks in chunks of max 100KB to preserve context
- Prioritizes: markdown cells > imports > function definitions > key analysis
- Skips: cell outputs, inline dataframes, verbose print statements
- Tracks cumulative token usage and summarizes older sections when needed

See `tests/test_notebook_reading_skill.py` for verification.

### Available Notebooks

| File | Size | Purpose |
|------|------|---------|
| `main.ipynb` | 3.6MB | Initial EDA, response distribution, feature correlations |
| `model_evaluation.ipynb` | 548KB | Grid search results, model comparisons (4 models) |
| `scoring_dancehall_tracks.ipynb` | 220KB | Apply trained models to dancehall tracks |

### Guidelines
- NEVER read entire notebooks at once
- Maximum 100KB per read operation
- Never include cell outputs (significant token overhead)
- Always summarize verbose sections to key observations

