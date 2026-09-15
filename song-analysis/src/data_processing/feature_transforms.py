# Handles adding new features to the base modelling dataset
# Should be created as a new column by applying to desired columns
from typing import Callable


def get_main_artist(artist_names: "list[str]") -> str:
    return artist_names.split("|")[0]

def condense_artist_attributes(artist_attrs: "list[str]", summ_func: Callable) -> "int":
    """
    Can pass `summ_func = max` for `max_aggr` or `summ_func = mean` for `mean_aggr`
    """
    proper_type_lst = [int(attr) for attr in artist_attrs.split("|")]
    agg_val = summ_func(proper_type_lst)

    return agg_val if isinstance(agg_val, int) else int(round(agg_val))

def get_release_year(date_str: str) -> int:
    """
    Given that dates are at most in YYYY-MM-DD format, we split by "-" and return the first value
    """
    return date_str.split("-")[0]