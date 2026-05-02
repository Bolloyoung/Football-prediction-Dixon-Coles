"""
football_data_uk.py
-------------------
Pull historical match data from www.football-data.co.uk — the canonical
free source for European league CSVs (since the early 1990s).

The site exposes one CSV per (league, season).  URL pattern:

    https://www.football-data.co.uk/mmz4281/<season-code>/<league-code>.csv

  season-code  e.g. 2526 = 2025-26, 2425 = 2024-25, 2324 = 2023-24
  league-code  E0  = Premier League
               E1  = Championship
               SP1 = La Liga
               D1  = Bundesliga
               I1  = Serie A
               F1  = Ligue 1

The CSVs include match results, full-time / half-time goals, shots, shots
on target, corners, fouls, yellow / red cards, and pre-match bookmaker
odds.  This is enough to build a rich training dataset for an ML model.

The functions degrade gracefully:
  * if the network is unavailable, they raise a clear `ScrapeError`
  * if a cached copy exists in `data/raw/` it is used unless `force=True`
"""

from __future__ import annotations

import os
import io
import datetime as dt
from typing import List, Optional

import pandas as pd
import requests


BASE_URL = "https://www.football-data.co.uk/mmz4281"

LEAGUE_CODES = {
    "EPL": "E0",            # English Premier League
    "Championship": "E1",
    "League_One":   "E2",
    "League_Two":   "E3",
    "LaLiga":       "SP1",
    "LaLiga2":      "SP2",
    "Bundesliga":   "D1",
    "Bundesliga2":  "D2",
    "SerieA":       "I1",
    "SerieB":       "I2",
    "Ligue1":       "F1",
    "Ligue2":       "F2",
    "Eredivisie":   "N1",
    "Primeira":     "P1",
    "ScottishPrem": "SC0",
    "Belgian":      "B1",
    "Greek":        "G1",
    "Turkish":      "T1",
}


class ScrapeError(RuntimeError):
    pass


def _season_code(start_year: int) -> str:
    """1996 -> '9697';  2025 -> '2526'."""
    a = start_year % 100
    b = (start_year + 1) % 100
    return f"{a:02d}{b:02d}"


def download_league_csv(league: str = "EPL",
                        start_year: int = 2024,
                        cache_dir: str = "data/raw",
                        force: bool = False,
                        timeout: int = 20) -> pd.DataFrame:
    """
    Download one season of one league.  Cache in `cache_dir`.

    Parameters
    ----------
    league : key from LEAGUE_CODES (e.g. 'EPL', 'LaLiga')
    start_year : season start year (2024 => 2024-25 season)
    cache_dir : where to store the downloaded CSV
    force : redownload even if cached
    """
    if league not in LEAGUE_CODES:
        raise ValueError(f"Unknown league {league!r}; pick from {list(LEAGUE_CODES)}")
    code  = LEAGUE_CODES[league]
    scode = _season_code(start_year)
    url   = f"{BASE_URL}/{scode}/{code}.csv"

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir, f"{league}_{scode}.csv")

    if os.path.exists(cache_path) and not force:
        try:
            return pd.read_csv(cache_path)
        except Exception:
            pass  # corrupt cache; re-download

    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise ScrapeError(f"failed to GET {url}: {e}") from e

    try:
        df = pd.read_csv(io.BytesIO(resp.content))
    except Exception as e:
        raise ScrapeError(f"failed to parse CSV from {url}: {e}") from e

    df.to_csv(cache_path, index=False)
    return df


def load_or_download(leagues: Optional[List[str]] = None,
                     start_year_from: int = 2014,
                     start_year_to:   Optional[int] = None,
                     cache_dir: str = "data/raw",
                     verbose: bool = False) -> pd.DataFrame:
    """
    Build a multi-league, multi-season match dataset.

    By default pulls the EPL since 2014-15.  Pass `leagues=['EPL','LaLiga',
    'Bundesliga','SerieA','Ligue1']` for a 5-big-leagues training corpus
    (~2k matches / season / league × 10 seasons ≈ 100k rows = "big enough").
    """
    leagues = leagues or ["EPL"]
    if start_year_to is None:
        start_year_to = dt.date.today().year

    frames: List[pd.DataFrame] = []
    for lg in leagues:
        for yr in range(start_year_from, start_year_to + 1):
            try:
                df = download_league_csv(lg, yr, cache_dir=cache_dir)
                df["League"]    = lg
                df["SeasonYr"]  = yr
                frames.append(df)
                if verbose:
                    print(f"  ✓ {lg} {yr}-{yr+1}: {len(df):4d} rows")
            except ScrapeError as e:
                if verbose:
                    print(f"  ✗ {lg} {yr}-{yr+1}: {e}")

    if not frames:
        raise ScrapeError("no data could be loaded for any (league, season)")

    full = pd.concat(frames, ignore_index=True, sort=False)

    # Standardise the most-used columns so downstream code is league-agnostic.
    rename_map = {
        "Date":   "date",   "HomeTeam": "home", "AwayTeam": "away",
        "FTHG":   "fthg",   "FTAG":     "ftag", "FTR":      "ftr",
        "HTHG":   "hthg",   "HTAG":     "htag", "HTR":      "htr",
        "HS":     "hs",     "AS":       "as_",  "HST":      "hst",
        "AST":    "ast",    "HF":       "hf",   "AF":       "af",
        "HC":     "hc",     "AC":       "ac",
        "HY":     "hy",     "AY":       "ay",   "HR":       "hr", "AR": "ar",
    }
    keep = ["date", "League", "SeasonYr", "home", "away",
            "fthg", "ftag", "ftr", "hs", "as_", "hst", "ast",
            "hf", "af", "hc", "ac", "hy", "ay", "hr", "ar"]
    out = full.rename(columns=rename_map)
    for col in keep:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[keep].copy()

    # Parse dates safely (UK CSVs use mixed dd/mm/yy and dd/mm/yyyy)
    out["date"] = pd.to_datetime(out["date"], dayfirst=True, errors="coerce")
    out = out.dropna(subset=["date", "home", "away", "fthg", "ftag"])
    out["fthg"] = out["fthg"].astype(int)
    out["ftag"] = out["ftag"].astype(int)
    out = out.sort_values("date").reset_index(drop=True)
    return out


__all__ = ["LEAGUE_CODES", "ScrapeError", "download_league_csv", "load_or_download"]
