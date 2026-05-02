"""
pipeline.py
-----------
High-level orchestrator that ties the live data pipeline together:

    scrape → train ML → fit Dixon-Coles → predict → blend → report

The Streamlit app and the tutorial notebook both call into this module
so there is exactly one source of truth for end-to-end behaviour.

Usage
~~~~~
    >>> from football_predictor.pipeline import run_full_pipeline
    >>> bundle = run_full_pipeline(
    ...     leagues=["EPL"], season_from=2014, season_to=2025,
    ...     model_path="models/epl.joblib", refresh=False
    ... )
    >>> bundle["predictions"]    # list[dict] for the next 14 days
"""

from __future__ import annotations

import os
import datetime as dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .scrapers import (
    load_or_download, fetch_upcoming_fixtures,
    fetch_league_xg, fetch_league_results,
)
from .scrapers.football_data_uk import ScrapeError as FDScrapeError
from .scrapers.understat       import ScrapeError as UScrapeError
from .scrapers.fixtures        import ScrapeError as FxScrapeError
from .ml_model import (
    build_feature_matrix, train_model, predict_match,
    save_bundle, load_bundle, blend_with_dixon_coles,
)
from .ratings  import AttackDefense, EloRating
from .model    import ScorelineDistribution


# --------------------------------------------------------------------------- #
#  Data acquisition                                                            #
# --------------------------------------------------------------------------- #

def acquire_history(leagues: List[str],
                    season_from: int,
                    season_to:   int,
                    cache_dir: str = "data/raw",
                    verbose: bool = True) -> pd.DataFrame:
    """Download + cache historical CSVs.  Returns a tidy match table."""
    return load_or_download(
        leagues=leagues,
        start_year_from=season_from,
        start_year_to=season_to,
        cache_dir=cache_dir,
        verbose=verbose,
    )


def acquire_xg(league: str, season: int) -> Optional[pd.DataFrame]:
    """Fetch live xG;  returns None on network failure."""
    try:
        return fetch_league_xg(league, season)
    except UScrapeError as e:
        print(f"[warn] understat unavailable: {e}")
        return None


def acquire_fixtures(league: str = "EPL",
                     window_days: int = 14) -> pd.DataFrame:
    """Upcoming fixtures within the next `window_days`."""
    today = dt.date.today()
    try:
        return fetch_upcoming_fixtures(
            league, from_date=today,
            to_date=today + dt.timedelta(days=window_days),
        )
    except FxScrapeError as e:
        print(f"[warn] fixturedownload unavailable: {e}")
        return pd.DataFrame(columns=["date", "home", "away", "kickoff"])


# --------------------------------------------------------------------------- #
#  Training (with caching)                                                     #
# --------------------------------------------------------------------------- #

def train_or_load(history: pd.DataFrame,
                  model_path: str,
                  refresh: bool = False) -> Tuple[object, Dict, Dict]:
    """
    If `model_path` exists and `refresh=False`, load it.  Otherwise build
    features, train, save, and return.
    """
    if os.path.exists(model_path) and not refresh:
        clf, meta = load_bundle(model_path)
        return clf, meta, {"loaded": model_path, "n_train": len(history)}

    X, y, meta = build_feature_matrix(history)
    clf, report = train_model(X, y, calibrate=True)
    save_bundle(clf, meta, model_path)
    return clf, meta, report


# --------------------------------------------------------------------------- #
#  Dixon-Coles fit from history                                                #
# --------------------------------------------------------------------------- #

def fit_dixon_coles_from_history(history: pd.DataFrame,
                                 last_season_only: bool = True
                                 ) -> Tuple[AttackDefense, EloRating, float]:
    """
    Fit Berrar attack/defense parameters and Elo from the historical
    match log.  Returns (AttackDefense, EloRating, league_mu).
    """
    df = history.copy()
    if last_season_only and "SeasonYr" in df.columns:
        max_yr = df["SeasonYr"].max()
        df = df[df["SeasonYr"] == max_yr]

    # ---- League mean goals per team-match ----
    if df.empty:
        mu = 1.4
    else:
        mu = float(((df["fthg"].sum() + df["ftag"].sum()) /
                    max(len(df) * 2, 1)))

    # ---- Aggregate per-team season stats ----
    teams = pd.unique(pd.concat([df["home"], df["away"]]))
    teams = [t for t in teams if isinstance(t, str)]
    rows = []
    for t in teams:
        h_mask = df["home"] == t
        a_mask = df["away"] == t
        played = int(h_mask.sum() + a_mask.sum())
        gf = int(df.loc[h_mask, "fthg"].sum() + df.loc[a_mask, "ftag"].sum())
        ga = int(df.loc[h_mask, "ftag"].sum() + df.loc[a_mask, "fthg"].sum())
        rows.append({"team": t, "played": played, "gf": gf, "ga": ga})
    season = pd.DataFrame(rows).set_index("team")

    # ---- Berrar closed-form attack / defence ----
    league_gf_pg = season["gf"].sum() / max(season["played"].sum(), 1)
    league_ga_pg = season["ga"].sum() / max(season["played"].sum(), 1)
    alpha, beta = {}, {}
    for t in season.index:
        played = max(int(season.loc[t, "played"]), 1)
        gf_pg = season.loc[t, "gf"] / played
        ga_pg = season.loc[t, "ga"] / played
        alpha[t] = gf_pg / max(league_gf_pg, 1e-6)
        beta[t]  = ga_pg / max(league_ga_pg, 1e-6)

    ad = AttackDefense(alpha=alpha, beta=beta, mu=mu)

    # ---- Elo from chronological history ----
    elo = EloRating()
    for r in df.itertuples():
        elo.update(r.home, r.away, int(r.fthg), int(r.ftag))

    return ad, elo, mu


# --------------------------------------------------------------------------- #
#  Prediction                                                                  #
# --------------------------------------------------------------------------- #

def predict_fixture(home: str, away: str,
                    ml_model, meta: Dict,
                    ad: AttackDefense, elo: EloRating,
                    blend_weight: float = 0.5,
                    kickoff: Optional[dt.date] = None) -> Dict:
    """
    Combine ML 1X2 with Dixon-Coles scoreline-derived 1X2.
    Returns a unified dict that mirrors `predict.predict_one`.
    """
    # ML
    ml = predict_match(ml_model, meta, home, away, kickoff)

    # Dixon-Coles base λ from attack × defence × home advantage
    lam_h, lam_a = ad.expected_goals(home, away)
    dist = ScorelineDistribution.from_rates(lam_h, lam_a)
    p_h, p_d, p_a = dist.outcome_probs()
    dc = {"home": p_h, "draw": p_d, "away": p_a}

    blended = blend_with_dixon_coles(ml, dc, w_ml=blend_weight)
    mh, ma, pm = dist.most_likely_scoreline()
    topN = dist.top_n_scorelines(8)

    return {
        "home": home, "away": away, "kickoff": kickoff,
        "lam_home": round(lam_h, 3), "lam_away": round(lam_a, 3),
        "p_home_ml": round(ml["home"], 4),
        "p_draw_ml": round(ml["draw"], 4),
        "p_away_ml": round(ml["away"], 4),
        "p_home_dc": round(dc["home"], 4),
        "p_draw_dc": round(dc["draw"], 4),
        "p_away_dc": round(dc["away"], 4),
        "p_home":    round(blended["home"], 4),
        "p_draw":    round(blended["draw"], 4),
        "p_away":    round(blended["away"], 4),
        "score":     f"{mh}-{ma}",
        "score_prob": round(pm, 4),
        "top_scorelines": topN,
        "over25":   round(dist.prob_over(2.5), 4),
        "btts":     round(dist.prob_btts(), 4),
        "elo_home": round(elo.ratings.get(home, 1500.0), 1),
        "elo_away": round(elo.ratings.get(away, 1500.0), 1),
    }


# --------------------------------------------------------------------------- #
#  Top-level orchestrator                                                      #
# --------------------------------------------------------------------------- #

def run_full_pipeline(leagues: List[str] = None,
                      season_from: int = 2014,
                      season_to:   Optional[int] = None,
                      model_path:  str = "models/epl.joblib",
                      refresh:     bool = False,
                      window_days: int = 14,
                      blend_weight: float = 0.5,
                      verbose:     bool = True) -> Dict:
    """
    Full data → ML → Dixon-Coles → predict pipeline.

    Returns a dict with:
      history, model_report, ad, elo, fixtures, predictions
    """
    leagues = leagues or ["EPL"]
    season_to = season_to or dt.date.today().year

    if verbose:
        print(f"[1/4] Acquiring history: {leagues}, "
              f"{season_from}-{season_from+1}…{season_to}-{season_to+1}")
    history = acquire_history(leagues, season_from, season_to, verbose=verbose)

    if verbose: print(f"[2/4] Training (or loading) ML model → {model_path}")
    ml_model, meta, report = train_or_load(history, model_path, refresh=refresh)

    if verbose: print(f"[3/4] Fitting Dixon-Coles attack/defence + Elo")
    ad, elo, mu = fit_dixon_coles_from_history(history, last_season_only=True)

    if verbose: print(f"[4/4] Pulling next-{window_days}-days fixtures")
    fixtures = acquire_fixtures(leagues[0], window_days)

    preds: List[Dict] = []
    for fx in fixtures.itertuples():
        try:
            p = predict_fixture(fx.home, fx.away, ml_model, meta, ad, elo,
                                blend_weight=blend_weight,
                                kickoff=fx.date if hasattr(fx, "date") else None)
            preds.append(p)
        except Exception as e:
            print(f"[warn] predict {fx.home} v {fx.away}: {e}")
    return {
        "history":   history,
        "model_report": report,
        "ad":        ad,
        "elo":       elo,
        "league_mu": mu,
        "fixtures":  fixtures,
        "predictions": preds,
        "model":     ml_model,
        "meta":      meta,
    }


__all__ = [
    "acquire_history", "acquire_xg", "acquire_fixtures",
    "train_or_load", "fit_dixon_coles_from_history",
    "predict_fixture", "run_full_pipeline",
]
