TARGET_AUDIO_FEATURE_LIST = [
    "danceability",
    "mode",
    "energy",
    "key",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
    "duration_ms",
]
TRACK_METADATA_FEATURE_LIST = [
    # "main_artist", # dropping as it inflates features without adding much value
    "release_year",
    # "max_aggr_popularity",
    "max_aggr_followers",  # assuming the song reaches all the audience of the larger artist
    # "mean_aggr_popularity",
    # "mean_aggr_followers",
]

TARGET_RESPONSE = "popularity"

AUDIO_FEATURE_DTYPE_MAP = {
    "mode": "category",
    "key": "category",
}
ALL_FEATURE_DTYPE_MAP = {
    **AUDIO_FEATURE_DTYPE_MAP,
    "release_year": "int64",
}

CATEGORICAL_COLS = ["mode", "key", "time_signature"]
