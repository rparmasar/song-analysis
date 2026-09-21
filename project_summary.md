# Song Analysis Project Summary

## tl;dr

The song analysis project built a machine learning model to predict Trinidad dancehall track popularity using hip-hop training data. We trained four models and selected **hist-gbm**, which achieved **RMSE = 12.5** and **R² = 0.44** on the test set.

Applied to dancehall tracks, the model predicts that **75% of tracks score below 43 popularity** with a maximum predicted score of **76** (hip-hop baseline max: 96). SHAP analysis identifies `max_aggr_followers` and `release_year` as the top predictors. Notably, many top-scoring artists from our December 2023 dataset—including Squash, Byron Messia, and Teejay—have since achieved global success, confirming the model's predictive validity for emerging talent.

---

## Overview

Predicts popularity of Trinidad dancehall tracks using a model trained on hip-hop audio features. Four models evaluated: elastic net, poisson glm, random forest, hist-gbm. Selected `hist-gbm` as best performer with strong generalization.

---

## Data Sources

Datasets sourced from [song-feature-extraction](https://github.com/rparmasar/song-feature-extraction) repository. Three CSV datasets in `song-analysis/data/`:

- `hip_hop-track-attributes.csv` (training data)
- `dancehall-hip-hop-track-attributes.csv`
- `trini_dancehall-track-attributes.csv` (scoring target)

Pull new data from repository before re-running for fresh results.

---

## Orchestrating the Pipeline

Single script entry point (`train_models.py`) that trains all four models. Loads hip-hop training data and applies common feature transforms (`create_common_features`). Splits dataset 80/20 with fixed seed (200294814) for reproducibility. Saves artifacts to `song-analysis/model_artifacts/all_features/`:

- Trained models: `elastic_net_model.joblib`, `poisson_glm_model.joblib`, `random_forest_model.joblib`, `hist_gbm_model.joblib`
- Splits: `train_features.csv`, `test_features.csv`, `train_response.csv`, `test_response.csv`
- Scaling artifacts: `fitted_scaler.joblib`, `fitted_encoder.joblib`
- Training metadata: `train_times_df.csv` (model training times and best params)

---

## Methodology

### Model Selection Approach
Compare RMSE and R² scores between train and test sets. Visual inspection of prediction vs. observed distributions. Lower RMSE + higher R² = better performance. No overfitting if train/test metrics are similar.

### Four Models Evaluated

#### 1. Elastic Net Linear Regression
- Combines LASSO and Ridge regularization
- Fastest training (~7s for 400 models)
- Similar train/test performance (not overfitting)
- Low R² indicates poor variance explanation

#### 2. Poisson GLM
- Assumes `popularity` is count-type variable
- Uses Poisson distribution vs. Normal in Elastic Net
- Slower training (~63s for 20 models)
- Similar performance to Elastic Net

#### 3. Random Forest with Poisson Criterion
- Ensemble of decision trees trained with Poisson loss
- Tuned hyperparameters: n_estimators, max_depth
- Slowest training (~304s for 200 models)
- Strong R² and RMSE on both train/test sets

#### 4. Hist-GBM (Histogram-Based Gradient Boosting)
- Uses histograms instead of individual samples
- Trains vertically with binning (faster than standard boosting)
- Inspired by LightGBM algorithm
- Very fast training (~66s for 600 models)
- Best test set performance with good generalization

### Best Model Selection
`hist-gbm` with all features selected as best model:
- Superior RMSE and R² on test set
- Minimal training time relative to Random Forest

### Feature Importance Insights
SHAP analysis reveals strongest predictors:
- `max_aggr_followers` (most important)
- `release_year` (second most important)
- Secondary features: speechiness, tempo, acousticness, liveness

Threshold insight: ~11K followers needed before popularity drops significantly.

---

## Scoring Dancehall Tracks

### Pre-processing Pipeline
Load fitted scaler and encoder from artifacts. Apply common feature transforms to dancehall data. Fix dtypes according to `ALL_FEATURE_DTYPE_MAP`. Encode categorical columns (`mode`, `key`, `time_signature`). Scale numeric features and combine encoded/scaled dataframes.

### Score Distribution Findings
- 75% of tracks scored below 43 popularity
- Maximum score: 76 (compared to 96 for hip-hop baseline)

### SHAP Interpretability Insights
- Penalized for low follower count (long blue tail on SHAP plot)
- Some high-follower tracks achieve higher expected popularity
- Release year favorable—dancehall not heavily penalized for older releases
- Genre characteristic: dancehall is relatively new genre
- Penalized for high speechiness, acousticness; low liveness

### Top 20 Tracks Analysis
- Majority released in 2023 (data collection year)
- Recency contributes to higher scores
- Most highly scored tracks have moderate follower counts (<200K)
- One track features collab with prominent US artist
- Predictive validity confirmed—many top artists gained global popularity post-Dec 2023

### Artists Who Gained Popularity
Squash, Byron Messia, Teejay, Skeng, Kraff Gad, Intence, Valiant, Tommy Lee Sparta, Medz Boss, Chronic Law

---

## Key Conclusions

- `hist-gbm` fitted on all features selected as best model
- Track metadata (followers and release year) are significant predictors of popularity
- Makes sense since popularity heavily influenced by Spotify's recommendation algorithm
- Dancehall tracks show different scoring patterns than hip-hop baseline
- Model demonstrates predictive validity for emerging artists
