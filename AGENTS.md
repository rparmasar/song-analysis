# Song Analysis Project Guide

## Overview

This repo analyzes Spotify track audio features to predict popularity of Trinidad dancehall tracks using a model trained on hip-hop data. It's implemented as a Python script with `main.py` at the root of `song-analysis/`.

## Entry Points

Run analysis from the parent directory:

```bash
python song-analysis/main.py
```

This reads data from `song-analysis/data/hip_hop-track-attributes.csv`, trains four models, and saves artifacts to `song-analysis/model_artifacts/`.

## Project Structure

```
song-analysis/
├── main.py                    # orchestration script
├── requirements.txt           # Python dependencies (scikit-learn, pandas, matplotlib, etc.)
├── data/                      # CSV datasets (hip_hop, dancehall hip-hop, trini_dancehall)
├── src/
│   ├── data_processing/
│   │   ├── __init__.py        # feature lists and dtype maps
│   │   ├── base_transforms.py # train/test split + encoding/scaling
│   │   └── feature_transforms.py # derived features (release_year, agg_popularity, etc.)
│   ├── model_training/
│   │   ├── fit_model.py       # grid search for ElasticNet, Poisson, RandomForest, HistGBM
│   │   └── evaluate_model.py  # model evaluation metrics
│   ├── visualization/         # plot training/test distributions, PDPs
│   └── types/                 # TrackAttributeColumns enum
├── model_artifacts/           # trained models, encoders, scalers, train/test splits
└── docs/next_steps.md         # TODO list
```

## Key Constants

Located in `src/data_processing/__init__.py`:

- `TARGET_RESPONSE = "popularity"`
- `TARGET_AUDIO_FEATURE_LIST` (14 audio features)
- `TRACK_METADATA_FEATURE_LIST` (release_year, max_aggr_followers)
- `CATEGORICAL_COLS = ["mode", "key", "time_signature"]`
- `AUDIO_FEATURE_DTYPE_MAP` / `ALL_FEATURE_DTYPE_MAP`

## Model Pipeline

1. Load CSV → 2. Add derived features → 3. Split/encode/scale (80/20, seed=200294814) → 4. Grid search models → 5. Save artifacts

### Models trained (all use 5-fold CV with RMSE scoring):

| Model | Search space |
|-------|--------------|
| ElasticNet | alpha, l1_ratio |
| PoissonRegressor | alpha |
| RandomForest | n_estimators (100-550), max_depth (4-8) |
| HistGradientBoostingRegressor | max_iter, max_features |

## Important Notes

### Data sources

Datasets are from [song-feature-extraction](https://github.com/rparmasar/song-feature-extraction). Always pull new data before re-running if you want fresh results.

### Artifacts

`joblib` files in `model_artifacts/` enable reproducibility without retraining. Don't modify these after training.

### Dancehall-specific warning

Dropping `main_artist` feature because it causes errors with dancehall tracks (see `feature_transforms.py:37-40`).

### Environment

The repo uses a virtual environment in `env/`. Activate before installing dependencies or running scripts.

## Common Commands

```bash
# Install dependencies (in the env)
pip install -r song-analysis/requirements.txt

# Run full pipeline
cd /path/to/song-analysis && python song-analysis/main.py

# After training, use loaded models to score new tracks via src/model_training/*.py
```

## Verification Checklist

Before considering a change "complete":

1. Data: Confirm correct CSV file in `data/` directory
2. Artifacts: Check `model_artifacts/` has expected `.joblib` files
3. Models: Verify `train_times_df.csv` exists with all four models
4. Reproducibility: Ensure seed=200294814 is used in any new splitting code

## Architecture Notes

- Uses `GridSearchCV` with 5-fold CV for all models
- Categorical features are one-hot encoded via `OneHotEncoder(handle_unknown="ignore")`
- Numerical features use `MinMaxScaler` after encoding
- All model fitting functions return `(model, train_time)` tuples for comparison