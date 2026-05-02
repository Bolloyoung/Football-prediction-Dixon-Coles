"""Data acquisition modules — historical CSVs, xG, and live fixtures."""

from .football_data_uk import (
    download_league_csv,
    load_or_download,
    LEAGUE_CODES,
)
from .understat import fetch_team_xg, fetch_league_xg, fetch_league_results
from .fixtures   import fetch_upcoming_fixtures, parse_kickoff
