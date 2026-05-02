"""
ratings.py
----------
Statistical rating systems for team strength.

Implements:
  * Elo ratings with goal-margin and home-advantage adjustments
  * Pi-ratings (separate home / away strengths)
  * Berrar-style attack/defense parameters fit from season totals
    (closed-form Maher / Dixon-Coles attack/defense coefficients)

All rating systems are exposed as simple Python classes so the rest of the
pipeline can pick any combination it wants as model features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .data import TEAMS, TeamSeason, HOME_ADVANTAGE, league_averages


# ----------------------------------------------------------------------
#  ELO
# ----------------------------------------------------------------------

@dataclass
class EloRating:
    """
    Elo rating with three football-specific extensions:
      1. Goal margin multiplier (weights bigger wins more)
      2. Home advantage offset added to the home team's rating for
         expectation calculations
      3. Time decay applied via the K-factor (larger K at season end)
    """
    base: float = 1500.0
    k: float = 20.0
    home_advantage: float = 80.0     # Elo points
    ratings: Dict[str, float] = field(default_factory=dict)

    def initialise(self, teams: List[str]) -> None:
        for t in teams:
            self.ratings.setdefault(t, self.base)

    def expectation(self, home: str, away: str) -> float:
        """Probability the home team wins (no draws) per Elo."""
        diff = (self.ratings[home] + self.home_advantage) - self.ratings[away]
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def update(self, home: str, away: str, gh: int, ga: int) -> None:
        """
        Update ratings after a completed match.  Score S = 1 home win,
        0.5 draw, 0 home loss.  Goal-margin multiplier follows FiveThirtyEight.
        """
        e_home = self.expectation(home, away)
        if gh > ga:
            s_home = 1.0
        elif gh == ga:
            s_home = 0.5
        else:
            s_home = 0.0
        margin = max(abs(gh - ga), 1)
        mult = ((margin + 1) ** 0.8) / (7.5 + 0.006 * abs(
            (self.ratings[home] + self.home_advantage) - self.ratings[away]))
        delta = self.k * mult * (s_home - e_home)
        self.ratings[home] += delta
        self.ratings[away] -= delta


# ----------------------------------------------------------------------
#  PI-RATINGS
# ----------------------------------------------------------------------

@dataclass
class PiRating:
    """
    Pi-rating variant (Constantinou & Fenton, 2013).  Keeps separate
    home and away strength values for each team.  A positive value means
    the team tends to win; a negative value means they tend to lose.
    """
    lambda_: float = 0.035      # home-learning rate
    gamma: float = 0.7          # away-to-home leak
    home: Dict[str, float] = field(default_factory=dict)
    away: Dict[str, float] = field(default_factory=dict)

    def initialise(self, teams: List[str]) -> None:
        for t in teams:
            self.home.setdefault(t, 0.0)
            self.away.setdefault(t, 0.0)

    def overall(self, team: str) -> float:
        return 0.5 * (self.home[team] + self.away[team])


# ----------------------------------------------------------------------
#  BERRAR-STYLE ATTACK/DEFENSE  (closed-form from season totals)
# ----------------------------------------------------------------------

@dataclass
class AttackDefense:
    """
    Attack and defense strengths (multiplicative) such that

        E[goals for team i vs team j at home] = alpha_i * beta_j * mu * gamma
        E[goals for team j vs team i at home] = alpha_j * beta_i * mu / gamma

    where:
        mu     = league average goals per team-game
        gamma  = home-advantage multiplier
        alpha  = attack strength (>1 attacks well)
        beta   = defense weakness (>1 concedes too much)

    We fit alpha_i from a team's per-match goals scored normalised by the
    league average, and beta_i from per-match goals conceded similarly.
    This is the Maher (1982) / Dixon-Coles closed-form approximation, which
    is accurate when teams play a roughly balanced schedule — a fair
    assumption inside one league late in the season.
    """
    alpha: Dict[str, float] = field(default_factory=dict)
    beta: Dict[str, float] = field(default_factory=dict)
    mu: float = 0.0
    gamma: float = HOME_ADVANTAGE

    @classmethod
    def fit(cls, teams: Dict[str, TeamSeason],
            xg_weight: float = 0.4) -> "AttackDefense":
        """
        Blend actual goals with expected goals to stabilise the estimate:
        effective_gf = (1 - w) * GF + w * xG, same for GA.

        xg_weight = 0.0  → pure goals
        xg_weight = 1.0  → pure xG
        Default 0.4 is a reasonable regularising blend recommended by
        the framework paper.
        """
        mu_team, _ = league_averages()

        alpha, beta = {}, {}
        for name, t in teams.items():
            eff_gf = (1 - xg_weight) * t.gf + xg_weight * t.xg
            eff_ga = (1 - xg_weight) * t.ga + xg_weight * t.xga
            gpg_for = eff_gf / t.mp
            gpg_against = eff_ga / t.mp
            alpha[name] = gpg_for / mu_team
            beta[name]  = gpg_against / mu_team
        return cls(alpha=alpha, beta=beta, mu=mu_team, gamma=HOME_ADVANTAGE)

    # ---- expected goals for a matchup ----

    def expected_goals(self, home: str, away: str,
                       form_home: float = 1.0,
                       form_away: float = 1.0) -> Tuple[float, float]:
        """
        Lambda (home) and Lambda (away) expected-goal rates for a single match.
        Recent form multiplies the attack only (defenders don't get 'hot form').
        """
        lam_h = self.alpha[home] * form_home * self.beta[away] * self.mu * self.gamma
        lam_a = self.alpha[away] * form_away * self.beta[home] * self.mu / self.gamma
        return lam_h, lam_a


# ----------------------------------------------------------------------
#  BUILD INITIAL RATINGS FROM SEASON RESULTS
# ----------------------------------------------------------------------

def build_elo_from_season(k: float = 20.0) -> EloRating:
    """
    Seeds Elo from each team's points-per-game: teams ±400 Elo per 1.0 PPG
    difference from the league mean of 1.33 PPG.  This is a rough seed;
    the proper use of Elo is to update match-by-match from the first game
    of the season — here we use it as a pre-computed strength prior.
    """
    elo = EloRating(k=k)
    elo.initialise(list(TEAMS.keys()))
    total_mp = sum(t.mp for t in TEAMS.values())
    total_pts = sum(t.points for t in TEAMS.values())
    league_ppg = total_pts / total_mp
    for name, t in TEAMS.items():
        ppg = t.points / t.mp
        elo.ratings[name] = 1500.0 + 400.0 * (ppg - league_ppg)
    return elo
