import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src.data_processing import CATEGORICAL_COLS, TARGET_RESPONSE

from ..types import TrackAttributeColumns


def get_modelling_dataset(
    filepath: str, dtypes_map: dict[TrackAttributeColumns,]
) -> pd.DataFrame:
    """
    NOTE: not used currently
    Loads in a dataframe from `filepath` and ensures the correct datatypes for columns.
    """
    # Read in Raw DataFrame
    raw_df = pd.read_csv(filepath, index_col=TrackAttributeColumns.ID.value)

    # Map dtypes to correct version using provided map
    modelling_df = raw_df.astype(dtypes_map)

    return modelling_df


def split_encode_and_scale_dataset(
    df: pd.DataFrame,
    response_col: str = TARGET_RESPONSE,
    cols_to_encode: list[str] = CATEGORICAL_COLS,
    test_set_ratio: float = 0.8,
    seed=22020020212314,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler, OneHotEncoder
]:
    """
    Splits the dataset into training and testing sets, then scales the features. Also returns the fitted scaler
    """
    # remove response from incoming df before splitting
    train_features, test_features, train_response, test_response = train_test_split(
        df.drop(response_col, axis=1),
        df[[response_col]],
        test_size=test_set_ratio,
        random_state=seed,
    )

    # apply encoding to categorical cols
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    encoded_train_features = encoder.fit_transform(train_features[cols_to_encode])
    encoded_test_features = encoder.transform(test_features[cols_to_encode])

    # convert above back to dataframes
    encoded_train_features_df = pd.DataFrame(
        encoded_train_features,
        columns=encoder.get_feature_names_out(),
        index=train_features.index,
    )
    encoded_test_features_df = pd.DataFrame(
        encoded_test_features,
        columns=encoder.get_feature_names_out(),
        index=test_features.index,
    )

    # fit scaling on train and apply on test (for numerical cols only)
    mm_scaler = MinMaxScaler()
    scaled_train_features = mm_scaler.fit_transform(
        train_features.drop(columns=cols_to_encode, axis=1)
    )
    scaled_test_features = mm_scaler.transform(
        test_features.drop(columns=cols_to_encode, axis=1)
    )

    # convert back to dataframes
    scaled_train_features_df = pd.DataFrame(
        scaled_train_features,
        columns=mm_scaler.get_feature_names_out(),
        index=train_features.index,
    )
    scaled_test_features_df = pd.DataFrame(
        scaled_test_features,
        columns=mm_scaler.get_feature_names_out(),
        index=test_features.index,
    )

    # reassemble
    train_features = pd.concat(
        [
            encoded_train_features_df,
            scaled_train_features_df,
        ],
        axis=1,
    )
    test_features = pd.concat(
        [encoded_test_features_df, scaled_test_features_df], axis=1
    )

    return (
        train_features,
        test_features,
        train_response,
        test_response,
        mm_scaler,
        encoder,
    )
