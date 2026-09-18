# Handles adding new features to the base modelling dataset
# Should be created as a new column by applying to desired columns
from collections.abc import Callable
from statistics import mean

import pandas as pd


def get_main_artist(artist_names: "list[str]") -> str:
    return artist_names.split("|")[0]


def condense_artist_attributes(artist_attrs: "list[str]", summ_func: Callable) -> "int":
    """
    Can pass `summ_func = max` for `max_aggr` or `summ_func = mean` for `mean_aggr`
    """
    proper_type_lst = [int(attr) for attr in artist_attrs.split("|")]
    agg_val = summ_func(proper_type_lst)

    return agg_val if isinstance(agg_val, int) else round(agg_val)


def get_release_year(date_str: str) -> int:
    """
    Given that dates are at most in YYYY-MM-DD format, we split by "-" and return the first value
    """
    return date_str.split("-")[0]


def create_common_features(track_df: pd.DataFrame) -> pd.DataFrame:
    """
    applies all the transforms above to the incoming `track_df` and onehot encodes the categorical columns
    """
    ## Add main_artist (replace $ with S to avoid errors)
    track_df.loc[:, "main_artist"] = track_df["track_artists_name"].apply(
        get_main_artist
    )
    track_df["main_artist"] = track_df["main_artist"].str.replace(
        r"\$", "S", regex=True
    )

    ## Add release_year
    track_df.loc[:, "release_year"] = track_df["release_date"].apply(get_release_year)

    ## Add max_aggr attributes
    track_df.loc[:, "max_aggr_popularity"] = track_df["track_artists_popularity"].apply(
        lambda val: condense_artist_attributes(val, max)
    )
    track_df.loc[:, "max_aggr_followers"] = track_df["track_artists_followers"].apply(
        lambda val: condense_artist_attributes(val, max)
    )

    ## Add mean_aggr attributes
    track_df.loc[:, "mean_aggr_popularity"] = track_df[
        "track_artists_popularity"
    ].apply(lambda val: condense_artist_attributes(val, mean))
    track_df.loc[:, "mean_aggr_followers"] = track_df["track_artists_followers"].apply(
        lambda val: condense_artist_attributes(val, mean)
    )

    return track_df
