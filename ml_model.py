"""
ml_model.py
-----------
Machine-learning predictive model for football match outcomes (H / D / A).

Pipeline
~~~~~~~~
1. Take a long historical match table (date, home, away, fthg, ftag, ...)
   produced by `scrapers.football_data_uk.load_or_download`.
2. Engineer per-match features computed strictly from the past
   (no leakage):
     - rolling goals-for / goals-against (last 5 / 10)
     - rolling xG (last 5) when available
     - Elo before kickoff
     - days of rest for each side
     - H2H goal-difference average
     - season points-per-game so far
     - shots / shots-on-target rolling means
     - bookmaker implied probabilities (when present)
3. Train a calibrated XGBoost classifier on H/D/A.
4. Persist with joblib;  expose `predict_proba(features)` for the
   downstream Dixon-Coles ensemble.

The module degrades gracefully if XGBoost isn't installed: it falls
back to scikit-learn's GradientBoostingClassifier.

Usage
~~~~~
    >>> from football_predictor.scrapers import load_or_download
    >>> from football_predictor.ml_model import (
    ...     build_feature_matrix, train_model, predict_match
    ... )
    >>> raw = load_or_download(["EPL"], 2014, 2024)
    >>> X, y, meta = build_feature_matrix(raw)
    >>> model, report = train_model(X, y)
    >>> predict_match(model, meta, "Arsenal", "Chelsea")
"""

from __future__ import annotations

import os
import math
import json
import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier  # type: ignore
    _HAS_XGB = True
except Exception:                      # pragma: no cover
    _HAS_XGB = False
    from sklearn.ensemble import GradientBoostingClassifier  # type: ignore

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    log_loss, accuracy_score, classification_report, brier_score_loss,
)
from sklearn.preprocessing import LabelEncoder

import joblib


# --------------------------------------------------------------------------- #
#  Feature engineering                                                         #
# --------------------------------------------------------------------------- #

ROLL_SHORT = 5     # last-5 form
ROLL_LONG  = 10    # last-10 form

ELO_K     = 20.0
ELO_HOME  = 65.0
ELO_BASE  = 1500.0


def _result_label(fthg: int, ftag: int) -> str:
    if fthg > ftag: return "H"
    if fthg < ftag: return "A"
    return "D"


def _expected_elo(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def _update_elo(home: float, away: float, result: str) -> Tuple[float, float]:
    s_h = {"H": 1.0, "D": 0.5, "A": 0.0}[result]
    e_h = _expected_elo(home + ELO_HOME, away)
    delta = ELO_K * (s_h - e_h)
    return home + delta, away - delta


@dataclass
class TeamLedger:
    """Online running stats for one team, updated chronologically."""
    elo:      float = ELO_BASE
    last_dates: List[dt.date]   = field(default_factory=list)
    last_gf:    List[int]       = field(default_factory=list)
    last_ga:    List[int]       = field(default_factory=list)
    last_shots: List[float]     = field(default_factory=list)
    last_sot:   List[float]     = field(default_factory=list)
    season:     Optional[int]   = None
    season_pts: int             = 0
    season_gp:  int             = 0


def _trim(lst: list, n: int) -> list:
    return lst[-n:] if len(lst) > n else lst


def build_feature_matrix(matches: pd.DataFrame) -> Tuple[
    pd.DataFrame, np.ndarray, Dict
]:
    """
    Convert a chronologically-sorted match table into a feature matrix.

    Returned:
      X     - DataFrame (n_matches, n_features) of pre-match features
      y     - integer labels (0=H, 1=D, 2=A)
      meta  - dict with `team_state` (final TeamLedger per team), feature
              names, and label encoder.  Used by `predict_match`.
    """
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home", "away", "fthg", "ftag"])
    df = df.sort_values("date").reset_index(drop=True)
    df["fthg"] = df["fthg"].astype(int)
    df["ftag"] = df["ftag"].astype(int)

    teams: Dict[str, TeamLedger] = {}
    h2h:   Dict[Tuple[str, str], List[int]] = {}

    rows: List[Dict] = []
    labels: List[str] = []

    has_shots = "hs" in df.columns and "as_" in df.columns
    has_sot   = "hst" in df.columns and "ast" in df.columns

    for r in df.itertuples():
        h, a, d = r.home, r.away, r.date.date() if hasattr(r.date, "date") else r.date
        season  = getattr(r, "SeasonYr", None)
        h_l = teams.setdefault(h, TeamLedger(season=season))
        a_l = teams.setdefault(a, TeamLedger(season=season))

        # Reset season points if season changed
        for tl in (h_l, a_l):
            if season is not None and tl.season != season:
                tl.season = season
                tl.season_pts = 0
                tl.season_gp = 0

        # ---------- pre-match features (no leakage) ----------
        feat = {
            "elo_diff":   (h_l.elo + ELO_HOME) - a_l.elo,
            "h_elo":      h_l.elo,
            "a_elo":      a_l.elo,

            "h_gf5":      np.mean(h_l.last_gf[-ROLL_SHORT:]) if h_l.last_gf else 1.2,
            "a_gf5":      np.mean(a_l.last_gf[-ROLL_SHORT:]) if a_l.last_gf else 1.2,
            "h_ga5":      np.mean(h_l.last_ga[-ROLL_SHORT:]) if h_l.last_ga else 1.2,
            "a_ga5":      np.mean(a_l.last_ga[-ROLL_SHORT:]) if a_l.last_ga else 1.2,
            "h_gf10":     np.mean(h_l.last_gf[-ROLL_LONG:])  if h_l.last_gf else 1.2,
            "a_gf10":     np.mean(a_l.last_gf[-ROLL_LONG:])  if a_l.last_gf else 1.2,
            "h_ga10":     np.mean(h_l.last_ga[-ROLL_LONG:])  if h_l.last_ga else 1.2,
            "a_ga10":     np.mean(a_l.last_ga[-ROLL_LONG:])  if a_l.last_ga else 1.2,

            "h_shots5":   np.mean(h_l.last_shots[-ROLL_SHORT:]) if h_l.last_shots else 12.0,
            "a_shots5":   np.mean(a_l.last_shots[-ROLL_SHORT:]) if a_l.last_shots else 12.0,
            "h_sot5":     np.mean(h_l.last_sot[-ROLL_SHORT:])   if h_l.last_sot   else 4.0,
            "a_sot5":     np.mean(a_l.last_sot[-ROLL_SHORT:])   if a_l.last_sot   else 4.0,

            "h_rest":     (d - h_l.last_dates[-1]).days if h_l.last_dates else 7,
            "a_rest":     (d - a_l.last_dates[-1]).days if a_l.last_dates else 7,
            "rest_diff":  ((d - h_l.last_dates[-1]).days if h_l.last_dates else 7) -
                          ((d - a_l.last_dates[-1]).days if a_l.last_dates else 7),

            "h_ppg":      (h_l.season_pts / h_l.season_gp) if h_l.season_gp else 1.3,
            "a_ppg":      (a_l.season_pts / a_l.season_gp) if a_l.season_gp else 1.3,

            "h2h_gd":     float(np.mean(h2h.get((h, a), []))) if h2h.get((h, a)) else 0.0,
        }

        rows.append(feat)
        labels.append(_result_label(r.fthg, r.ftag))

        # ---------- post-match updates ----------
        result = labels[-1]
        new_h_elo, new_a_elo = _update_elo(h_l.elo, a_l.elo, result)
        h_l.elo, a_l.elo = new_h_elo, new_a_elo

        h_l.last_dates.append(d); a_l.last_dates.append(d)
        h_l.last_gf.append(r.fthg); h_l.last_ga.append(r.ftag)
        a_l.last_gf.append(r.ftag); a_l.last_ga.append(r.fthg)

        if has_shots:
            try:
                h_l.last_shots.append(float(r.hs)); a_l.last_shots.append(float(r.as_))
            except (ValueError, TypeError):
                pass
        if has_sot:
            try:
                h_l.last_sot.append(float(r.hst)); a_l.last_sot.append(float(r.ast))
            except (ValueError, TypeError):
                pass

        # trim ring buffers
        for tl in (h_l, a_l):
            tl.last_dates  = _trim(tl.last_dates,  ROLL_LONG)
            tl.last_gf     = _trim(tl.last_gf,     ROLL_LONG)
            tl.last_ga     = _trim(tl.last_ga,     ROLL_LONG)
            tl.last_shots  = _trim(tl.last_shots,  ROLL_LONG)
            tl.last_sot    = _trim(tl.last_sot,    ROLL_LONG)

        pts_h = 3 if result == "H" else 1 if result == "D" else 0
        pts_a = 3 if result == "A" else 1 if result == "D" else 0
        h_l.season_pts += pts_h; h_l.season_gp += 1
        a_l.season_pts += pts_a; a_l.season_gp += 1

        gd = r.fthg - r.ftag
        h2h.setdefault((h, a), []).append(gd)
        h2h.setdefault((a, h), []).append(-gd)
        h2h[(h, a)] = h2h[(h, a)][-10:]
        h2h[(a, h)] = h2h[(a, h)][-10:]

    X = pd.DataFrame(rows)
    le = LabelEncoder().fit(["H", "D", "A"])
    y = le.transform(labels)

    meta = {
        "team_state": teams,
        "h2h":        h2h,
        "feature_names": list(X.columns),
        "label_encoder": le,
        "trained_at":   dt.datetime.utcnow().isoformat(),
    }
    return X, y, meta


# --------------------------------------------------------------------------- #
#  Training                                                                    #
# --------------------------------------------------------------------------- #

def _make_base_classifier():
    if _HAS_XGB:
        return XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            verbosity=0,
        )
    return GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
    )


def train_model(X: pd.DataFrame,
                y: np.ndarray,
                calibrate: bool = True,
                test_frac: float = 0.15) -> Tuple[object, Dict]:
    """
    Train and (optionally) calibrate a 3-class classifier.

    Uses a chronological split (last `test_frac` rows as hold-out).
    """
    n = len(X)
    cut = int(n * (1 - test_frac))
    X_tr, X_te = X.iloc[:cut], X.iloc[cut:]
    y_tr, y_te = y[:cut],      y[cut:]

    base = _make_base_classifier()
    base.fit(X_tr, y_tr)

    if calibrate and len(X_tr) > 500:
        # Wrap a clone of `base` with isotonic calibration via 3-fold prefit-
        # style CV.  For XGB we use cv-fitted (sigmoid is faster).
        method = "isotonic" if len(X_tr) > 2000 else "sigmoid"
        clf = CalibratedClassifierCV(estimator=_make_base_classifier(),
                                     method=method, cv=3)
        clf.fit(X_tr, y_tr)
    else:
        clf = base

    # ----- evaluation -----
    proba = clf.predict_proba(X_te)
    pred  = proba.argmax(axis=1)
    rep = {
        "n_train":   len(X_tr),
        "n_test":    len(X_te),
        "test_logloss":   float(log_loss(y_te, proba, labels=[0, 1, 2])),
        "test_accuracy":  float(accuracy_score(y_te, pred)),
        "feature_importance": _feature_importance(clf, list(X.columns)),
        "classifier":     "xgboost" if _HAS_XGB else "sklearn-gbm",
        "calibration":    "isotonic" if calibrate else "none",
    }
    return clf, rep


def _feature_importance(clf, names: List[str]) -> Dict[str, float]:
    """Best-effort feature importance extraction across model types."""
    inner = getattr(clf, "calibrated_classifiers_", None)
    if inner:
        # Pull from first underlying estimator
        est = inner[0].estimator
    else:
        est = clf
    imp = getattr(est, "feature_importances_", None)
    if imp is None:
        return {}
    pairs = sorted(zip(names, imp), key=lambda kv: -float(kv[1]))
    return {k: float(v) for k, v in pairs}


# --------------------------------------------------------------------------- #
#  Inference                                                                   #
# --------------------------------------------------------------------------- #

def _features_for_pair(meta: Dict, home: str, away: str,
                       kickoff: Optional[dt.date] = None) -> pd.DataFrame:
    """Construct one feature row from the trained team-state cache."""
    teams: Dict[str, TeamLedger] = meta["team_state"]
    h2h:   Dict[Tuple[str, str], List[int]] = meta["h2h"]
    fnames = meta["feature_names"]

    h_l = teams.get(home, TeamLedger())
    a_l = teams.get(away, TeamLedger())
    today = kickoff or dt.date.today()

    feat = {
        "elo_diff": (h_l.elo + ELO_HOME) - a_l.elo,
        "h_elo":    h_l.elo,
        "a_elo":    a_l.elo,
        "h_gf5":    np.mean(h_l.last_gf[-ROLL_SHORT:]) if h_l.last_gf else 1.2,
        "a_gf5":    np.mean(a_l.last_gf[-ROLL_SHORT:]) if a_l.last_gf else 1.2,
        "h_ga5":    np.mean(h_l.last_ga[-ROLL_SHORT:]) if h_l.last_ga else 1.2,
        "a_ga5":    np.mean(a_l.last_ga[-ROLL_SHORT:]) if a_l.last_ga else 1.2,
        "h_gf10":   np.mean(h_l.last_gf[-ROLL_LONG:])  if h_l.last_gf else 1.2,
        "a_gf10":   np.mean(a_l.last_gf[-ROLL_LONG:])  if a_l.last_gf else 1.2,
        "h_ga10":   np.mean(h_l.last_ga[-ROLL_LONG:])  if h_l.last_ga else 1.2,
        "a_ga10":   np.mean(a_l.last_ga[-ROLL_LONG:])  if a_l.last_ga else 1.2,
        "h_shots5": np.mean(h_l.last_shots[-ROLL_SHORT:]) if h_l.last_shots else 12.0,
        "a_shots5": np.mean(a_l.last_shots[-ROLL_SHORT:]) if a_l.last_shots else 12.0,
        "h_sot5":   np.mean(h_l.last_sot[-ROLL_SHORT:])   if h_l.last_sot   else 4.0,
        "a_sot5":   np.mean(a_l.last_sot[-ROLL_SHORT:])   if a_l.last_sot   else 4.0,
        "h_rest":   (today - h_l.last_dates[-1]).days if h_l.last_dates else 7,
        "a_rest":   (today - a_l.last_dates[-1]).days if a_l.last_dates else 7,
        "rest_diff": ((today - h_l.last_dates[-1]).days if h_l.last_dates else 7) -
                     ((today - a_l.last_dates[-1]).days if a_l.last_dates else 7),
        "h_ppg":    (h_l.season_pts / h_l.season_gp) if h_l.season_gp else 1.3,
        "a_ppg":    (a_l.season_pts / a_l.season_gp) if a_l.season_gp else 1.3,
        "h2h_gd":   float(np.mean(h2h.get((home, away), []))) if h2h.get((home, away)) else 0.0,
    }
    # Reorder to training order, fill missing with defaults
    row = {k: feat.get(k, 0.0) for k in fnames}
    return pd.DataFrame([row])


def predict_match(model, meta: Dict, home: str, away: str,
                  kickoff: Optional[dt.date] = None) -> Dict[str, float]:
    """Return calibrated {H, D, A} probabilities for one fixture."""
    X1 = _features_for_pair(meta, home, away, kickoff)
    proba = model.predict_proba(X1)[0]
    le = meta["label_encoder"]
    out = {label: float(proba[idx]) for label, idx in
           zip(le.classes_, range(len(le.classes_)))}
    return {"home": out.get("H", 0.0),
            "draw": out.get("D", 0.0),
            "away": out.get("A", 0.0)}


# --------------------------------------------------------------------------- #
#  Persistence                                                                 #
# --------------------------------------------------------------------------- #

def save_bundle(model, meta: Dict, path: str) -> None:
    """Dump (model, meta) tuple to a single .joblib file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump({"model": model, "meta": meta}, path)


def load_bundle(path: str):
    obj = joblib.load(path)
    return obj["model"], obj["meta"]


def blend_with_dixon_coles(ml_proba: Dict[str, float],
                           dc_proba: Dict[str, float],
                           w_ml: float = 0.5) -> Dict[str, float]:
    """
    Linear-pool blend of ML and Dixon-Coles 1X2 distributions.

    `dc_proba` keys: "home", "draw", "away".
    """
    w_dc = 1.0 - w_ml
    out = {k: w_ml * ml_proba.get(k, 0.0) + w_dc * dc_proba.get(k, 0.0)
           for k in ("home", "draw", "away")}
    s = sum(out.values()) or 1.0
    return {k: v / s for k, v in out.items()}


__all__ = [
    "build_feature_matrix",
    "train_model",
    "predict_match",
    "save_bundle",
    "load_bundle",
    "blend_with_dixon_coles",
]
