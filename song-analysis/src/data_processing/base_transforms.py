import pandas as pd

from ..types import TrackAttributeColumns

def get_modelling_dataset(filepath: str, dtypes_map: dict[TrackAttributeColumns, ]) -> pd.DataFrame:
    """
    Loads in a dataframe from `filepath` and ensures the correct datatypes for columns.
    """
    # Read in Raw DataFrame
    raw_df = pd.read_csv(filepath, index_col=TrackAttributeColumns.ID.value)

    # Map dtypes to correct version using provided map
    modelling_df = raw_df.astype(dtypes_map)

    return modelling_df