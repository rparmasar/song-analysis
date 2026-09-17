from pathlib import Path

import joblib
import pandas as pd
from src.data_processing import (
    FEATURE_DTYPE_MAP,
    TARGET_AUDIO_FEATURE_LIST,
    TARGET_RESPONSE,
    TRACK_METADATA_FEATURE_LIST,
)
from src.data_processing.base_transforms import split_encode_and_scale_dataset
from src.data_processing.feature_transforms import create_common_features
from src.model_training.fit_model import (
    fit_elastic_net,
    fit_hist_gbm,
    fit_poisson_glm,
    fit_random_forest,
)


def main():
    # some path constants (change between audio and all features)
    BASE_MODEL_ARTIFACTS_PATH = Path("song-analysis") / "model_artifacts"
    MODEL_TYPE = "all_features"
    MODEL_ARTIFACTS_PATH = BASE_MODEL_ARTIFACTS_PATH / MODEL_TYPE

    # figure out which features based on model type
    RELEVANT_COLS = (
        [
            *TARGET_AUDIO_FEATURE_LIST,
            *TRACK_METADATA_FEATURE_LIST,
            TARGET_RESPONSE,
        ]
        if MODEL_TYPE == "all_features"
        else [*TARGET_AUDIO_FEATURE_LIST, TARGET_RESPONSE]
    )

    # load data
    track_df = pd.read_csv(
        filepath_or_buffer="song-analysis/data/hip_hop-track-attributes.csv",
        index_col="track_id",
    )

    print(f"[main] loaded base dataset with {track_df.shape=}")

    # apply common feature transforms
    base_modelling_df = create_common_features(track_df)

    print(
        f"[main] created features to produce dataset with shape {base_modelling_df.shape=}"
    )

    # subset columns and apply type transforms
    modelling_df = base_modelling_df[RELEVANT_COLS]
    modelling_df = modelling_df.astype(FEATURE_DTYPE_MAP)

    print(
        f"[main] finished pre-processing dataset with shape {modelling_df.shape=} and the following full list of features - {list(modelling_df.columns)}"
    )

    # split into train/test split and apply scaling
    (
        train_features,
        test_features,
        train_response,
        test_response,
        mm_scaler,
        encoder,
    ) = split_encode_and_scale_dataset(
        df=modelling_df,
        test_set_ratio=0.2,
        seed=200294814,
    )
    print(
        f"[main] split, encoded and scaled data into train/test with shapes {train_features.shape=}, {test_features.shape=}, {train_response.shape=}, {test_response.shape=}"
    )

    # save scaler and encoder for later use
    joblib.dump(mm_scaler, f"{MODEL_ARTIFACTS_PATH}/fitted_scaler.joblib")
    print(
        f"[main] saved fitted scaler to disk at {MODEL_ARTIFACTS_PATH}/fitted_scaler.joblib"
    )
    joblib.dump(encoder, f"{MODEL_ARTIFACTS_PATH}/fitted_encoder.joblib")
    print(
        f"[main] saved fitted encoder to disk at {MODEL_ARTIFACTS_PATH}/fitted_encoder.joblib"
    )

    # model fitting (all features)
    ## elastic net
    all_features_elastic_net_model, all_features_elastic_net_model_train_time = (
        fit_elastic_net(
            train_features=train_features,
            train_response=train_response,
        )
    )

    ## poisson glm
    all_features_poisson_glm, all_features_poisson_glm_train_time = fit_poisson_glm(
        train_features=train_features,
        train_response=train_response,
    )

    ## random forest
    all_features_random_forest_model, all_features_random_forest_model_train_time = (
        fit_random_forest(
            train_features=train_features,
            train_response=train_response,
        )
    )

    ## histogram-based gbm
    all_features_hist_gbm, all_features_hist_gbm_train_time = fit_hist_gbm(
        train_features=train_features,
        train_response=train_response,
    )

    ## save these to evaluate in notebook
    ALL_FEATURES_MODELS = {
        "elastic_net": all_features_elastic_net_model,
        "poisson_glm": all_features_poisson_glm,
        "random_forest": all_features_random_forest_model,
        "hist_gbm": all_features_hist_gbm,
    }

    train_time_df = pd.DataFrame(
        {
            "model": ALL_FEATURES_MODELS.keys(),
            "train_time": [
                all_features_elastic_net_model_train_time,
                all_features_poisson_glm_train_time,
                all_features_random_forest_model_train_time,
                all_features_hist_gbm_train_time,
            ],
        }
    )

    train_time_df.to_csv(MODEL_ARTIFACTS_PATH / "train_times_df.csv", index=False)

    for model_name, model in ALL_FEATURES_MODELS.items():
        joblib.dump(model, MODEL_ARTIFACTS_PATH / f"{model_name}_model.joblib")
        print(
            f"[main] saved {model_name=} to {MODEL_ARTIFACTS_PATH / f'{model_name}_model.joblib'}"
        )

    print(
        f"[main] finished training models and saving artifacts. final training times: {train_time_df}"
    )


if __name__ == "__main__":
    main()
