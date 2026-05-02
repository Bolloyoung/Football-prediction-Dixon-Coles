"""
fixtures.py
-----------
Lightweight upcoming-fixture scraper.  Sites used (in order of preference):

    1. fixturedownload.com — clean CSVs, public, no auth.
    2. football-data.co.uk fixtures CSV (when it exists).

Both endpoints can break; the function returns an empty DataFrame on
failure rather than crashing the pipeline so the rest of the system
keeps working with hand-edited fixtures.
"""

from __future__ import annotations

import io
import datetime as dt
from typing import Optional

import pandas as pd
import requests

UA = "Mozilla/5.0 (compatible; football-predictor/1.0)"

FIXTUREDOWNLOAD_URLS = {
    "EPL": "https://fixturedownload.com/download/epl-2025-UTC.csv",
    "Championship": "https://fixturedownload.com/download/champ-2025-UTC.csv",
    "LaLiga":    "https://fixturedownload.com/download/laliga-2025-UTC.csv",
    "SerieA":    "https://fixturedownload.com/download/seriea-2025-UTC.csv",
    "Bundesliga":"https://fixturedownload.com/download/bundesliga-2025-UTC.csv",
    "Ligue1":    "https://fixturedownload.com/download/ligue1-2025-UTC.csv",
}


class ScrapeError(RuntimeError):
    pass


def parse_kickoff(s: str) -> Optional[dt.datetime]:
    """Parse 'dd/mm/yyyy hh:mm' or ISO-ish strings safely."""
    if not isinstance(s, str): return None
    for fmt in ("%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M",
                "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        return pd.to_datetime(s, dayfirst=True, errors="coerce").to_pydatetime()
    except Exception:
        return None


def fetch_upcoming_fixtures(league: str = "EPL",
                            from_date: Optional[dt.date] = None,
                            to_date:   Optional[dt.date] = None,
                            cache_dir: str = "data/raw",
                            timeout: int = 20) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        date, home, away, kickoff
    Filtered to fixtures kicking off in [from_date, to_date].
    """
    url = FIXTUREDOWNLOAD_URLS.get(league)
    if not url:
        raise ValueError(f"no fixturedownload url for {league!r}")

    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
    except (requests.RequestException, pd.errors.ParserError) as e:
        raise ScrapeError(f"fetch_upcoming_fixtures({league}): {e}") from e

    rename = {"Date":     "kickoff_str",
              "Home Team": "home",
              "Away Team": "away",
              "Round Number": "round"}
    df = df.rename(columns=rename)
    if "kickoff_str" not in df.columns:
        raise ScrapeError("fixture CSV missing 'Date' column")
    df["kickoff"] = df["kickoff_str"].apply(parse_kickoff)
    df["date"]    = df["kickoff"].apply(lambda v: v.date() if v else None)

    today = dt.date.today()
    from_date = from_date or today
    to_date   = to_date   or (today + dt.timedelta(days=14))
    mask = df["date"].between(from_date, to_date, inclusive="both")
    out = df.loc[mask, ["date", "home", "away", "kickoff"]].copy()
    return out.sort_values("kickoff").reset_index(drop=True)


__all__ = ["fetch_upcoming_fixtures", "parse_kickoff", "ScrapeError"]
