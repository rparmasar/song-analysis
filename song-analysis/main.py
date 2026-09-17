from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from src.data_processing import (
    FEATURE_DTYPE_MAP,
    TARGET_AUDIO_FEATURE_LIST,
    TARGET_RESPONSE,
    TRACK_METADATA_FEATURE_LIST,
)
from src.data_processing.base_transforms import split_and_scale_dataset
from src.data_processing.feature_transforms import create_common_features
from src.model_training.fit_model import fit_elastic_net


def main():
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
    RELEVANT_COLS = [
        *TARGET_AUDIO_FEATURE_LIST,
        *TRACK_METADATA_FEATURE_LIST,
        TARGET_RESPONSE,
    ]
    modelling_df = base_modelling_df[RELEVANT_COLS]
    modelling_df = modelling_df.astype(FEATURE_DTYPE_MAP)

    print(
        f"[main] finished pre-processing dataset with shape {modelling_df.shape=} and the following full list of features - {list(modelling_df.columns)}"
    )

    # split into train/test split and apply scaling
    (
        base_train_features,
        base_test_features,
        train_response,
        test_response,
        mm_scaler,
    ) = split_and_scale_dataset(
        df=modelling_df,
        test_set_ratio=0.8,
        seed=200294814,
    )
    print(
        f"[main] split data into base train/test with shapes {base_train_features.shape=}, {base_test_features.shape=}, {train_response.shape=}, {test_response.shape=}"
    )

    # now apply one-hot encoding
    encoder = OneHotEncoder(handle_unknown="ignore")
    train_features = encoder.fit_transform(base_train_features)
    test_features = encoder.transform(base_test_features)

    print(
        f"[main] fitted onehot encoder and transformed base train/test data with shapes {train_features.shape=}, {test_features.shape=}"
    )

    # save scaler and encoder for later use
    BASE_MODEL_ARTIFACTS_PATH = Path("song-analysis") / "model_artifacts"
    joblib.dump(mm_scaler, f"{BASE_MODEL_ARTIFACTS_PATH}/fitted_scaler.joblib")
    print(
        f"[main] saved fitted scaler to disk at {BASE_MODEL_ARTIFACTS_PATH}/fitted_scaler.joblib"
    )
    joblib.dump(encoder, f"{BASE_MODEL_ARTIFACTS_PATH}/fitted_encoder.joblib")
    print(
        f"[main] saved fitted encoder to disk at {BASE_MODEL_ARTIFACTS_PATH}/fitted_encoder.joblib"
    )

    # model fitting (all features)
    ## elastic net
    all_features_elastic_net_model = fit_elastic_net(
        train_features=train_features,
        train_response=train_response,
    )


if __name__ == "__main__":
    main()
