"""
features.py
-----------
Context-feature engine that adjusts base Poisson rates (lambda_home,
lambda_away) using the "niche" factors called out in the framework:

    1. Head-to-Head with exponential time decay
    2. Player discipline (yellows / reds / fouls)
    3. Rest days (fixture congestion)
    4. Travel fatigue (away team only)
    5. Referee card-tendency × team discipline
    6. Set-piece efficiency × opponent weakness on set-pieces
    7. Clean-sheet rate → defensive regularisation
    8. Key-player absence (injury_index)
    9. Match importance (title / relegation race)

Each helper returns an *adjustment factor* (typically in [0.7, 1.3]) that
multiplies the attacker's or defender's expected-goal rate.  The adjustments
combine multiplicatively, and we cap the product in a reasonable range so a
stack of small factors can't blow up the prediction.

Every adjustment is also returned as a labelled dict so the visualisation
layer can show exactly *why* the model leans a certain way (a mini
"contribution plot" serving the same role SHAP would in a full XGBoost
implementation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .data import (
    TEAMS, H2H, REFEREES,
    REFEREE_BASE_CARDS, TODAY,
    TeamSeason, Fixture,
)


# ----------------------------------------------------------------------
#  HEAD-TO-HEAD
# ----------------------------------------------------------------------

def h2h_goal_deltas(home: str, away: str,
                    half_life_days: int = 365) -> Tuple[float, float, int]:
    """
    Compute the exponentially-decayed home / away average goals for this
    fixture over past meetings.

    Returns (home_avg, away_avg, matches_used).  matches_used==0 → no data.
    """
    numer_h = numer_a = denom = 0.0
    count = 0
    for date, h, a, gh, ga in H2H:
        teams = {h, a}
        if {home, away} != teams:
            continue
        age = (TODAY - date).days
        w   = 0.5 ** (age / half_life_days)
        # put goals in the perspective of the requested home team
        if h == home:
            numer_h += w * gh
            numer_a += w * ga
        else:
            numer_h += w * ga
            numer_a += w * gh
        denom  += w
        count  += 1
    if denom == 0:
        return 0.0, 0.0, 0
    return numer_h / denom, numer_a / denom, count


def h2h_adjustment(home: str, away: str,
                   strength: float = 0.12) -> Tuple[float, float]:
    """
    Compare decayed H2H goal rates to both teams' season average and nudge
    lambda toward whichever side has historically dominated.

    `strength` controls how aggressive the nudge is; 0.12 = max ±12 %.
    """
    h_avg, a_avg, used = h2h_goal_deltas(home, away)
    if used == 0:
        return 1.0, 1.0

    home_gpg = TEAMS[home].gf / TEAMS[home].mp
    away_gpg = TEAMS[away].gf / TEAMS[away].mp

    # relative delta vs season average
    d_home = (h_avg - home_gpg) / max(home_gpg, 0.5)
    d_away = (a_avg - away_gpg) / max(away_gpg, 0.5)

    # squash to ±strength
    mult_h = 1.0 + strength * math.tanh(d_home)
    mult_a = 1.0 + strength * math.tanh(d_away)
    return mult_h, mult_a


# ----------------------------------------------------------------------
#  DISCIPLINE  (reds inflate goals-conceded, dampen goals-scored)
# ----------------------------------------------------------------------

def discipline_adjustment(team: TeamSeason,
                          ref_cards: float = REFEREE_BASE_CARDS,
                          strength: float = 0.05) -> Tuple[float, float]:
    """
    Return (attack_mult, defense_boost_on_opponent) for one team.

    A team that picks up a lot of reds loses attacking impetus (attack × <1)
    and concedes more on transition (opponent's attack ×>1).  A harsh
    referee amplifies the effect.
    """
    ref_ratio = ref_cards / REFEREE_BASE_CARDS
    # Expected reds per match (scaled by referee).
    reds = team.reds_pg * ref_ratio

    # Red cards cost ~0.15 goals scored / ~0.25 goals conceded on average.
    attack_mult = 1.0 - strength * reds * 1.50
    opp_attack_boost = 1.0 + strength * reds * 2.50
    return max(0.80, attack_mult), min(1.25, opp_attack_boost)


# ----------------------------------------------------------------------
#  REST DAYS  /  FATIGUE
# ----------------------------------------------------------------------

def rest_adjustment(days: int) -> float:
    """
    Fatigue multiplier on a team's attack rate.
      ≤2 days: 0.90 (heavy fatigue)
      3 days:  0.95
      4 days:  0.98
      ≥5 days: 1.00
      ≥10 days: 1.02 (rested but slightly rusty — neutralised)
    """
    if days <= 2:   return 0.90
    if days == 3:   return 0.95
    if days == 4:   return 0.98
    if days <= 9:   return 1.00
    return 1.02


# ----------------------------------------------------------------------
#  TRAVEL
# ----------------------------------------------------------------------

def travel_adjustment(km: int) -> float:
    """
    Away-team fatigue.  Short trips (<250 km) are neutral; very long
    (>500 km, cross-country) shave ~3 %.
    """
    if km <= 200:   return 1.00
    if km <= 350:   return 0.99
    if km <= 500:   return 0.98
    return 0.97


# ----------------------------------------------------------------------
#  SET PIECES
# ----------------------------------------------------------------------

def set_piece_adjustment(attacker: TeamSeason,
                         defender: TeamSeason,
                         strength: float = 0.06) -> float:
    """
    Teams that score a big share from set pieces exploit weak defenders
    (proxy: defender's clean-sheet rate & injury_index which tends to hit
    set-piece marking first).
    """
    att_share  = attacker.set_piece_share
    def_weak   = (1 - defender.clean_sheet_rate) + 0.5 * defender.injury_index
    # centred so average teams multiply by ~1.0
    bump = strength * (att_share - 0.30) * (def_weak - 0.70) / 0.10
    return max(0.95, min(1.08, 1.0 + bump))


# ----------------------------------------------------------------------
#  KEY-PLAYER ABSENCE
# ----------------------------------------------------------------------

def injury_adjustment(team: TeamSeason,
                      attack_strength: float = 0.35,
                      defense_strength: float = 0.20) -> Tuple[float, float]:
    """
    Returns (attack_mult, own_defence_mult).  Heavy absences drop both.
    """
    idx = max(0.0, min(1.0, team.injury_index))
    return (1.0 - attack_strength * idx,
            1.0 - defense_strength * idx)


# ----------------------------------------------------------------------
#  MATCH IMPORTANCE
# ----------------------------------------------------------------------

def importance_adjustment(imp: float) -> float:
    """
    High-importance matches tend to be tighter (scores compressed).
    We scale total expected goals by a factor slightly <1 for very high
    stakes (tightening, like a Champions League final effect).
    """
    if imp >= 1.3:  return 0.94
    if imp >= 1.2:  return 0.97
    if imp >= 1.1:  return 0.99
    return 1.00


# ----------------------------------------------------------------------
#  MASTER COMBINER
# ----------------------------------------------------------------------

@dataclass
class AdjustmentBreakdown:
    name: str
    home_factor: float
    away_factor: float
    note: str = ""


def combine_adjustments(fx: Fixture) -> Tuple[float, float, List[AdjustmentBreakdown]]:
    """
    Compute multiplicative adjustments on (home attack rate, away attack rate)
    for a single fixture.  Returns (home_mult, away_mult, breakdown_list).
    """
    home = TEAMS[fx.home]
    away = TEAMS[fx.away]
    ref_cards = REFEREES.get(fx.referee, REFEREE_BASE_CARDS)

    parts: List[AdjustmentBreakdown] = []

    # 1. H2H
    h2h_h, h2h_a = h2h_adjustment(fx.home, fx.away)
    parts.append(AdjustmentBreakdown("H2H history", h2h_h, h2h_a,
                                     f"weighted by {365}-day half-life"))

    # 2. Discipline
    a_h, d_boost_a = discipline_adjustment(home, ref_cards)  # home's cards help away
    a_a, d_boost_h = discipline_adjustment(away, ref_cards)
    # home attack multiplier = home_own_attack * home_benefits_from_away_reds
    disc_h = a_h * d_boost_h
    disc_a = a_a * d_boost_a
    parts.append(AdjustmentBreakdown(
        "Discipline × Ref", disc_h, disc_a,
        f"referee={fx.referee} ({ref_cards:.1f} cards/m)"))

    # 3. Rest days
    rest_h = rest_adjustment(fx.rest_home)
    rest_a = rest_adjustment(fx.rest_away)
    parts.append(AdjustmentBreakdown(
        "Rest days", rest_h, rest_a,
        f"home={fx.rest_home}d  away={fx.rest_away}d"))

    # 4. Travel (away only)
    trav_a = travel_adjustment(fx.travel_km)
    parts.append(AdjustmentBreakdown("Travel (away)", 1.0, trav_a,
                                     f"{fx.travel_km} km"))

    # 5. Set pieces
    sp_h = set_piece_adjustment(home, away)
    sp_a = set_piece_adjustment(away, home)
    parts.append(AdjustmentBreakdown(
        "Set-piece edge", sp_h, sp_a,
        "attacker's share vs defender's frailty"))

    # 6. Injuries
    inj_h, own_def_h = injury_adjustment(home)
    inj_a, own_def_a = injury_adjustment(away)
    # own_defence_mult reduces opponent's attack (they find it easier): invert
    opp_h_boost = 2.0 - own_def_a     # if own_def_a < 1, opp scores more
    opp_a_boost = 2.0 - own_def_h
    inj_home = inj_h * opp_h_boost / 1.0   # home attack × boost from away injuries
    inj_away = inj_a * opp_a_boost / 1.0
    parts.append(AdjustmentBreakdown(
        "Injuries", inj_home, inj_away,
        f"home={home.injury_index:.2f}  away={away.injury_index:.2f}"))

    # 7. Match importance
    imp = importance_adjustment(fx.importance)
    parts.append(AdjustmentBreakdown("Importance", imp, imp,
                                     f"importance={fx.importance:.2f}"))

    # combine
    home_mult = h2h_h * disc_h * rest_h * 1.0  * sp_h * inj_home * imp
    away_mult = h2h_a * disc_a * rest_a * trav_a * sp_a * inj_away * imp

    # clip the final product
    home_mult = max(0.65, min(1.45, home_mult))
    away_mult = max(0.65, min(1.45, away_mult))

    return home_mult, away_mult, parts
