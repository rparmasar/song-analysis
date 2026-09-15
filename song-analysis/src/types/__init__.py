from enum import Enum


class TrackAttributeColumns(Enum):
    """
    For further desciptions, see [Spotify API | get-several-audio-features](https://developer.spotify.com/documentation/web-api/reference/get-several-audio-features)
    """
    ID: str = 'track_id'
    NAME: str = 'track_name'
    ARTISTS_IDS: str = 'artist_ids'
    ARTISTS_NAMES: str = 'artist_names'
    ARTISTS_FOLLOWERS: int = 'artist_followers'
    ARTISTS_GENRES: int = 'artist_genres'
    ARTISTS_POPULARITY: int = 'artist_popularity'
    DANCEABILITY: float = 'danceability'
    ENERGY: float = 'energy'
    KEY: int = 'key'
    LOUDNESS: float = 'loudness'
    MODE: int = 'mode'
    SPEECHINESS: float = 'speechiness'
    ACOUSTICNESS: float = 'acousticness'
    INSTRUMENTALNESS: float = 'instrumentalness'
    LIVENESS: float = 'liveness'
    VALENCE: float = 'valence'
    TEMPO: float = 'temp'
    DURATION_MS: int = 'duration_ms'
    TIME_SIGNATURE: int = 'time_signature'
    RELEASE_DATE: str = 'release_date'
    RELEASE_DATE_PRECISION: str = 'release_date_precision'
    POPULARITY: int = 'popularity'
    GENERATED_TIMESTAMP: str = 'generated_timestamp'