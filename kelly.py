"""
kelly.py
--------
Kelly-criterion stake-sizing and +EV filter.

Given a model probability p and decimal bookmaker odds o:
  b = o - 1
  f* = (b * p - (1 - p)) / b      (Kelly fraction; clamped to 0 below)

Expected value per 1 unit:  EV = p * (o - 1) - (1 - p)

We provide:
  * kelly_fraction(p, odds)      — raw Kelly fraction
  * expected_value(p, odds)      — expected value per 1 unit stake
  * filter_positive_ev(preds)    — keep only bets where EV > 0
"""

from __future__ import annotations

from typing import Dict, List


def kelly_fraction(p: float, decimal_odds: float, fraction: float = 1.0) -> float:
    """
    Return the stake as a fraction of bankroll.

    fraction=1.0  → full Kelly (aggressive)
    fraction=0.5  → half Kelly (common in practice)
    fraction=0.25 → quarter Kelly (conservative)
    """
    if decimal_odds <= 1.0:
        return 0.0
    b = decimal_odds - 1.0
    f = (b * p - (1 - p)) / b
    return max(0.0, fraction * f)


def expected_value(p: float, decimal_odds: float) -> float:
    return p * (decimal_odds - 1.0) - (1.0 - p)


def filter_positive_ev(candidates: List[Dict],
                       min_edge: float = 0.02) -> List[Dict]:
    """
    Keep only picks with a positive edge above `min_edge` (default 2 %).
    Each candidate dict must expose:
        prob          — model probability
        decimal_odds  — bookmaker decimal odds
    The function adds edge, kelly_full, kelly_half keys.
    """
    kept = []
    for c in candidates:
        ev = expected_value(c["prob"], c["decimal_odds"])
        if ev >= min_edge:
            c["edge"] = ev
            c["kelly_full"] = kelly_fraction(c["prob"], c["decimal_odds"], 1.0)
            c["kelly_half"] = kelly_fraction(c["prob"], c["decimal_odds"], 0.5)
            kept.append(c)
    return kept
