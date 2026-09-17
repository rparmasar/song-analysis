import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.data_processing import TARGET_RESPONSE

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


def split_and_scale_dataset(
    df: pd.DataFrame,
    response_col: str = TARGET_RESPONSE,
    test_set_ratio: float = 0.8,
    seed=22020020212314,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into training and testing sets, then scales the features.
    """
    # remove response from incoming df before splitting
    train_features, test_features, train_response, test_response = train_test_split(
        df.drop(response_col, axis=1),
        df[[response_col]],
        test_size=test_set_ratio,
        random_state=seed,
    )

    # fit scaling on train and apply on test
    mm_scaler = MinMaxScaler()
    train_features_scaled = mm_scaler.fit_transform(train_features)

    test_features_scaled = mm_scaler.transform(test_features)

    return train_features_scaled, test_features_scaled, train_response, test_response
