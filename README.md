# song analysis

- [song analysis](#song-analysis)
  - [tl;dr](#tldr)
  - [Overview](#overview)
  - [Data Sources](#data-sources)
  - [How to Run](#how-to-run)
    - [What This Does](#what-this-does)
    - [Expected Outputs](#expected-outputs)
    - [Next Steps: Exploring Results](#next-steps-exploring-results)
    - [Data Refresh](#data-refresh)
  - [Methodology](#methodology)
    - [Model Selection Approach](#model-selection-approach)
    - [Four Models Evaluated](#four-models-evaluated)
      - [1. Elastic Net Linear Regression](#1-elastic-net-linear-regression)
      - [2. Poisson GLM](#2-poisson-glm)
      - [3. Random Forest with Poisson Criterion](#3-random-forest-with-poisson-criterion)
      - [4. Hist-GBM (Histogram-Based Gradient Boosting)](#4-hist-gbm-histogram-based-gradient-boosting)
    - [Best Model Selection](#best-model-selection)
    - [Feature Importance Insights](#feature-importance-insights)
  - [Scoring Dancehall Tracks](#scoring-dancehall-tracks)
    - [Pre-processing Pipeline](#pre-processing-pipeline)
    - [Score Distribution Findings](#score-distribution-findings)
    - [SHAP Interpretability Insights](#shap-interpretability-insights)
    - [Top 20 Tracks Analysis](#top-20-tracks-analysis)
    - [Artists Who Gained Popularity](#artists-who-gained-popularity)
  - [Key Conclusions](#key-conclusions)
  - [Limitations](#limitations)
  - [Improvements](#improvements)


## tl;dr

The song analysis project built a machine learning model to predict Trinidad dancehall track popularity using hip-hop training data, trained four models and selected **hist-gbm**, which achieved **RMSE = 12.5** and **R² = 0.44** on the test set.

Applied to dancehall tracks, the model predicts that **75% of tracks score below 43 popularity** with a maximum predicted score of **76** (hip-hop baseline max: 96). SHAP analysis identifies `max_aggr_followers` and `release_year` as the top predictors. 

Notably, many top-scoring artists from our December 2023 dataset—including Squash, Byron Messia, Kraff, Tommy Lee Sparta, and Teejay—have since achieved global success, confirming the model's predictive validity for emerging talent.


## Overview

Predicts popularity of Trinidad dancehall tracks using a model trained on hip-hop audio features. Four models evaluated: elastic net, poisson glm, random forest, hist-gbm. Selected `hist-gbm` as best performer with strong generalization.


## Data Sources

Datasets sourced from [song-feature-extraction](https://github.com/rparmasar/song-feature-extraction) repository. Three CSV datasets in `song-analysis/data/`:

- `hip_hop-track-attributes.csv` (training data)
- `trini_dancehall-track-attributes.csv` (scoring target)
- `dancehall-hip-hop-track-attributes.csv` (unused but a combined dataset of the other two)

Pull new data from repository before re-running for fresh results. *(requires Spotify Premium subscription)*


## How to Run

**Quick Start:** Activate your virtual environment, install dependencies, then run:

```bash
source env/bin/activate
pip install -r requirements.txt
python song-analysis/train_models.py
```

### What This Does

The script runs the full pipeline automatically:

1. **Loads Data**: Reads `hip_hop-track-attributes.csv` from the data folder
2. **Feature Engineering**: Creates common derived features (release_year, max_aggr_followers)
3. **Train/Test Split**: 80/20 split with fixed seed (200294814) for reproducibility
4. **Model Training**: Trains all four models with grid search (5-fold CV)
5. **Artifact Saving**: Saves trained models, splits, scalers, and encoder to `model_artifacts/all_features/`

### Expected Outputs

After running the script, you'll have:

- **Trained Models**: `elastic_net_model.joblib`, `poisson_glm_model.joblib`, `random_forest_model.joblib`, `hist_gbm_model.joblib`
- **Data Splits**: `train_features.csv`, `test_features.csv`, `train_response.csv`, `test_response.csv`
- **Scalers/Encoders**: `fitted_scaler.joblib`, `fitted_encoder.joblib`
- **Training Metadata**: `train_times_df.csv` with training times and best hyperparameters

### Next Steps: Exploring Results

After running the script, use these notebooks to explore the results:

1. **[model_evaluation.ipynb](./docs/model_evaluation.ipynb)** - Compare all four models, view prediction vs observed plots, analyze feature importance with SHAP
2. **[scoring_dancehall_tracks.ipynb](./docs/scoring_dancehall_tracks.ipynb)** - Apply the trained `hist-gbm` model to dancehall tracks, see score distributions, and explore top predictions

### Data Refresh

To re-run with fresh data:

1. Pull latest data from [song-feature-extraction](https://github.com/rparmasar/song-feature-extraction)
2. Re-run the script (new artifacts will overwrite old ones)
3. Note: Requires Spotify Premium subscription


Loads hip-hop training data and applies common feature transforms (`create_common_features`). Splits dataset 80/20 with fixed seed (200294814) for reproducibility. Saves artifacts to `song-analysis/model_artifacts/all_features/`:

- Trained models: `elastic_net_model.joblib`, `poisson_glm_model.joblib`, `random_forest_model.joblib`, `hist_gbm_model.joblib`
- Splits: `train_features.csv`, `test_features.csv`, `train_response.csv`, `test_response.csv`
- Scaling artifacts: `fitted_scaler.joblib`, `fitted_encoder.joblib`
- Training metadata: `train_times_df.csv` (model training times and best params)


## Methodology

### Model Selection Approach
Compare RMSE and R² scores between train and test sets + visual inspection of prediction vs. observed distributions. 

Lower RMSE + higher R² = better performance.

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


## Key Conclusions

- `hist-gbm` fitted on all features selected as best model
- Track metadata (followers and release year) are significant predictors of popularity
- Makes sense since popularity heavily influenced by Spotify's recommendation algorithm
- Dancehall tracks show different scoring patterns than hip-hop baseline
- Model demonstrates predictive validity for emerging artists

## Limitations

- The data was pulled on January 8, 2024, which may not reflect the current trends in both the dancehall and hip-hop genres.

- The audio features used in modelling likely come from Spotify's internal models and so there is a risk of error compounding since we don't have a concrete understanding of how those features are generated.

## Improvements

- Re-doing this analysis with fresh data would be beneficial to ensure the results are more accurate and see if the assumption that hip-hop tracks can predict popularity still holds true amidst changes in the genre landscape.

- Can explore creating new features using the more powerful LLMs of today, e.g.:

    * Speech-to-Text models for generating lyrics (and then NLP techniques like embeddings or simpler bag of words approach to extract features).

    * Can also explore Genius to fetch lyrics.

    * Passing audio files to models and prompting for feature extraction.

    
