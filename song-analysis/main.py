import pandas as pd
from src.data_processing import (
    FEATURE_DTYPE_MAP,
    TARGET_AUDIO_FEATURE_LIST,
    TARGET_RESPONSE,
    TRACK_METADATA_FEATURE_LIST,
)
from src.data_processing.base_transforms import split_and_scale_dataset
from src.data_processing.feature_transforms import create_common_features


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

    # subset columns, apply type transforms and apply one-hot encoding
    RELEVANT_COLS = [
        *TARGET_AUDIO_FEATURE_LIST,
        *TRACK_METADATA_FEATURE_LIST,
        TARGET_RESPONSE,
    ]
    modelling_df = base_modelling_df[RELEVANT_COLS]
    modelling_df = modelling_df.astype(FEATURE_DTYPE_MAP)
    modelling_df = pd.get_dummies(modelling_df)

    print(
        f"[main] finished pre-processing dataset with shape {modelling_df.shape=} and the following full list of features - {list(modelling_df.columns)}"
    )

    # split into train/test split and apply scaling
    train_features, test_features, train_response, test_response = (
        split_and_scale_dataset(
            df=modelling_df,
            test_set_ratio=0.8,
            seed=200294814,
        )
    )
    print(
        f"[main] split data into train/test with shapes {train_features.shape=}, {test_features.shape=}, {train_response.shape=}, {test_response.shape=}"
    )

    # model fitting


if __name__ == "__main__":
    main()
