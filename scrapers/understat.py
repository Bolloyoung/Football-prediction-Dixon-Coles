"""
understat.py
------------
Light-weight scraper for understat.com.

Understat embeds its data as JSON inside `<script>JSON.parse('...')</script>`
blocks on each page. We pull the page, locate the embedded blob, decode it,
and return clean Python dicts / DataFrames.

Three helpers:

    fetch_league_xg(league, season)        ── full xG table for a season
    fetch_team_xg(team_url_slug, season)   ── per-team match log with xG
    fetch_league_results(league, season)   ── all results with xG

If understat is unreachable, a `ScrapeError` is raised; callers should
fall back to whatever cached data they have.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from typing import Dict, List, Optional

import pandas as pd
import requests


BASE = "https://understat.com"
LEAGUE_PATHS = {
    "EPL":          "league/EPL",
    "LaLiga":       "league/La_liga",
    "Bundesliga":   "league/Bundesliga",
    "SerieA":       "league/Serie_A",
    "Ligue1":       "league/Ligue_1",
    "RFPL":         "league/RFPL",
}

UA = "Mozilla/5.0 (compatible; football-predictor/1.0)"


class ScrapeError(RuntimeError):
    pass


def _get(url: str, timeout: int = 20) -> str:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
        r.raise_for_status()
        return r.text
    except requests.RequestException as e:
        raise ScrapeError(f"GET {url}: {e}") from e


def _extract_json(html: str, var_name: str) -> List[Dict]:
    """
    Understat embeds data as `var teamsData = JSON.parse('...')`.
    Pull the JSON.parse argument, unescape it, and decode.
    """
    pat = re.compile(
        r"var\s+" + re.escape(var_name) +
        r"\s*=\s*JSON\.parse\('([^']+)'\)"
    )
    m = pat.search(html)
    if not m:
        raise ScrapeError(f"variable {var_name!r} not found")
    raw = m.group(1).encode("utf-8").decode("unicode_escape")
    return json.loads(raw)


def fetch_league_xg(league: str = "EPL",
                    season: int = 2025) -> pd.DataFrame:
    """
    Fetch the season-aggregate team xG table for `league`.
    `season` is the start year of the season (2025 => 2025-26).
    """
    if league not in LEAGUE_PATHS:
        raise ValueError(f"unknown league {league!r}; "
                         f"pick from {list(LEAGUE_PATHS)}")
    url = f"{BASE}/{LEAGUE_PATHS[league]}/{season}"
    html = _get(url)

    # teamsData is a dict keyed by team_id; each team has 'history' = list
    # of per-match dicts.
    data = _extract_json(html, "teamsData")
    if isinstance(data, dict):
        teams = list(data.values())
    else:
        teams = list(data)

    rows = []
    for team in teams:
        name = team.get("title")
        hist = team.get("history", [])
        if not hist:
            continue
        agg = {
            "team":      name,
            "matches":   len(hist),
            "wins":      sum(1 for h in hist if h.get("wins") == 1),
            "draws":     sum(1 for h in hist if h.get("draws") == 1),
            "losses":    sum(1 for h in hist if h.get("loses") == 1),
            "goals":     sum(int(h.get("scored", 0))   for h in hist),
            "ga":        sum(int(h.get("missed", 0))   for h in hist),
            "xG":        round(sum(float(h.get("xG", 0.0))  for h in hist), 2),
            "xGA":       round(sum(float(h.get("xGA", 0.0)) for h in hist), 2),
            "xPts":      round(sum(float(h.get("xpts", 0.0)) for h in hist), 2),
            "ppda":      round(sum(_safe_div(d.get("ppda", {})) for d in hist) /
                               max(len(hist), 1), 2),
        }
        rows.append(agg)
    df = pd.DataFrame(rows).sort_values("xG", ascending=False).reset_index(drop=True)
    return df


def _safe_div(d) -> float:
    try:
        a = float(d.get("att", 0)); b = float(d.get("def", 0))
        return a / b if b else 0.0
    except Exception:
        return 0.0


def fetch_team_xg(team_url_slug: str, season: int = 2025) -> pd.DataFrame:
    """
    Per-match xG history for one team.

    `team_url_slug` is the Understat URL slug, e.g. 'Manchester_City',
    'Arsenal', 'Liverpool'.  Browse understat.com to confirm.
    """
    url = f"{BASE}/team/{urllib.parse.quote(team_url_slug)}/{season}"
    html = _get(url)
    data = _extract_json(html, "datesData")
    df = pd.DataFrame(data)
    if df.empty:
        return df
    # flatten nested `xG`, `xGA`, `goals` dicts
    for col in ("xG", "xGA", "goals"):
        if col in df.columns and df[col].apply(lambda v: isinstance(v, dict)).any():
            df[col + "_h"] = df[col].apply(lambda v: float(v.get("h", 0)) if isinstance(v, dict) else None)
            df[col + "_a"] = df[col].apply(lambda v: float(v.get("a", 0)) if isinstance(v, dict) else None)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


def fetch_league_results(league: str = "EPL",
                         season: int = 2025) -> pd.DataFrame:
    """
    Every match in a season with goals + xG (home and away).
    """
    if league not in LEAGUE_PATHS:
        raise ValueError(f"unknown league {league!r}")
    url  = f"{BASE}/{LEAGUE_PATHS[league]}/{season}"
    html = _get(url)
    data = _extract_json(html, "datesData")
    rows = []
    for m in data:
        if m.get("isResult") is False:
            continue
        rows.append({
            "date":  m.get("datetime"),
            "home":  m["h"]["title"] if isinstance(m.get("h"), dict) else m.get("h"),
            "away":  m["a"]["title"] if isinstance(m.get("a"), dict) else m.get("a"),
            "fthg":  int(m["goals"]["h"]) if isinstance(m.get("goals"), dict) else None,
            "ftag":  int(m["goals"]["a"]) if isinstance(m.get("goals"), dict) else None,
            "xG_h":  float(m["xG"]["h"])  if isinstance(m.get("xG"),    dict) else None,
            "xG_a":  float(m["xG"]["a"])  if isinstance(m.get("xG"),    dict) else None,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["fthg", "ftag"]).sort_values("date").reset_index(drop=True)
    return df


__all__ = [
    "LEAGUE_PATHS", "ScrapeError",
    "fetch_league_xg", "fetch_team_xg", "fetch_league_results",
]
