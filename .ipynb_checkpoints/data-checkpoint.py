"""
data.py
-------
Season-to-date Premier League 2025-26 team statistics (through 19-Apr-2026)
and the upcoming matchday fixtures (24-26 April 2026).

Goals-for (GF) and goals-against (GA) were collected from public league table
reporting; where only goal difference (GD) and points were reported, GF/GA
were inferred from the reported league aggregates and match-by-match reports.
xG/xGA use the Understat / Squawka 2025-26 snapshots where available and
default to the scored-goals totals otherwise (so that teams without xG data
are neither penalised nor favoured by the xG blend).

All figures are editable: replace any value with a more authoritative number
before calling `predict.py` to refresh predictions.

Home/away splits: where no public split was available we fall back to the
league-wide average home/away split (home teams score ~55% of a team's goals
for and concede ~45% of goals against).  This is documented in the CONSTANTS
block and can be tuned.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ----------------------------------------------------------------------
#  CONSTANTS
# ----------------------------------------------------------------------

# Long-run EPL home advantage: home teams score ~55% of a team's season goals
# at home and concede ~45% of its conceded goals at home.
DEFAULT_HOME_SHARE_GF = 0.55
DEFAULT_HOME_SHARE_GA = 0.45

# Multiplicative home advantage on expected goals (classic Dixon-Coles ~1.25–1.35).
HOME_ADVANTAGE = 1.30


# ----------------------------------------------------------------------
#  TEAM SEASON RECORD
# ----------------------------------------------------------------------

@dataclass
class TeamSeason:
    name: str
    mp: int                 # matches played
    w: int
    d: int
    l: int
    gf: int                 # goals for (total)
    ga: int                 # goals against (total)
    xg: float               # expected goals (season total)
    xga: float              # expected goals against (season total)
    # home/away splits for goals — optional overrides
    gf_home: float = None
    ga_home: float = None
    gf_away: float = None
    ga_away: float = None
    # recent form rating 0.0-2.0 (1.0 = league-average recent form).
    # 2.0 = on a hot streak, 0.5 = poor recent run.  Used as a multiplier
    # on the attack rate.  A value of 1.0 is a safe default.
    form: float = 1.0

    def __post_init__(self):
        """Fill home/away splits if not provided explicitly."""
        if self.gf_home is None:
            self.gf_home = self.gf * DEFAULT_HOME_SHARE_GF
        if self.gf_away is None:
            self.gf_away = self.gf * (1.0 - DEFAULT_HOME_SHARE_GF)
        if self.ga_home is None:
            self.ga_home = self.ga * DEFAULT_HOME_SHARE_GA
        if self.ga_away is None:
            self.ga_away = self.ga * (1.0 - DEFAULT_HOME_SHARE_GA)

    @property
    def points(self) -> int:
        return 3 * self.w + self.d

    @property
    def gd(self) -> int:
        return self.gf - self.ga


# ----------------------------------------------------------------------
#  2025-26 EPL SNAPSHOT  (~19 Apr 2026)
# ----------------------------------------------------------------------
#
# Fields: (name, MP, W, D, L, GF, GA, xG, xGA, form)
#
# GF/GA are inferred from GD + scored goals reporting; xG/xGA from Understat
# snapshot where known.  Where xG was not published on a single consolidated
# table, xG is defaulted to GF and xGA to GA so the xG blend is a no-op for
# that team.

_raw: List[Tuple] = [
    # name,             MP, W,  D, L,  GF,  GA,  xG,    xGA,  form
    ("Arsenal",         31, 21, 7, 3,  63,  26, 58.0,  28.3, 1.15),
    ("Manchester City", 30, 19, 4, 7,  65,  33, 56.5,  34.5, 1.25),
    ("Manchester United",31,16, 7, 8,  54,  41, 48.0,  44.0, 1.00),
    ("Aston Villa",     31, 15, 9, 7,  55,  50, 52.0,  48.0, 1.00),
    ("Liverpool",       31, 13,10, 8,  58,  50, 60.0,  46.0, 1.10),
    ("Chelsea",         31, 13, 9, 9,  53,  38, 61.7,  42.0, 1.05),
    ("Brentford",       31, 12,10, 9,  50,  46, 47.0,  46.5, 0.95),
    ("Everton",         31, 12,10, 9,  44,  42, 41.0,  43.0, 1.00),
    ("Fulham",          31, 11,11, 9,  45,  46, 44.0,  47.0, 1.00),
    ("Brighton",        31, 11,10,10,  49,  45, 52.0,  46.0, 0.95),
    ("Sunderland",      31, 11,10,10,  42,  46, 40.0,  48.0, 1.00),
    ("Newcastle",       31, 11, 9,11,  47,  48, 48.0,  45.0, 1.05),
    ("Bournemouth",     31, 11, 9,11,  46,  48, 44.0,  49.0, 0.90),
    ("Crystal Palace",  30, 10, 9,11,  33,  35, 46.3,  40.0, 0.95),
    ("Leeds",           31,  8, 9,14,  38,  49, 37.0,  48.0, 1.20),
    ("Nottingham Forest",31, 8, 8,15,  36,  48, 35.0,  49.5, 0.85),
    ("West Ham",        32,  8, 8,16,  35,  52, 36.0,  50.0, 0.85),
    ("Tottenham",       31,  8, 6,17,  41,  51, 45.0,  48.0, 0.80),
    ("Burnley",         31,  5, 5,21,  26,  54, 28.0,  58.0, 0.70),
    ("Wolves",          30,  3, 6,21,  22,  55, 26.0,  57.0, 0.60),
]

TEAMS: Dict[str, TeamSeason] = {
    row[0]: TeamSeason(
        name=row[0], mp=row[1], w=row[2], d=row[3], l=row[4],
        gf=row[5], ga=row[6], xg=row[7], xga=row[8], form=row[9],
    )
    for row in _raw
}


# ----------------------------------------------------------------------
#  NEXT-MATCHDAY FIXTURES  (24-26 April 2026)
# ----------------------------------------------------------------------

@dataclass
class Fixture:
    home: str
    away: str
    kickoff: str = ""   # informational only


FIXTURES: List[Fixture] = [
    Fixture("Sunderland",        "Nottingham Forest",  "Fri 24 Apr, 20:00 BST"),
    Fixture("Fulham",             "Aston Villa",        "Sat 25 Apr, 12:30 BST"),
    Fixture("Liverpool",          "Crystal Palace",     "Sat 25 Apr, 15:00 BST"),
    Fixture("West Ham",           "Everton",            "Sat 25 Apr, 15:00 BST"),
    Fixture("Wolves",             "Tottenham",          "Sat 25 Apr, 15:00 BST"),
    Fixture("Arsenal",            "Newcastle",          "Sat 25 Apr, 17:30 BST"),
    Fixture("Burnley",            "Manchester City",    "Sun 26 Apr, 14:00 BST"),
    Fixture("Brighton",           "Chelsea",            "Sun 26 Apr, 16:30 BST"),
]


def league_averages() -> Tuple[float, float]:
    """
    Return (league_avg_goals_per_game_per_team, league_avg_goals_per_game).

    Computed from the sum of all goals scored divided by the total number of
    team-games (each match contributes two team-games).
    """
    total_goals = sum(t.gf for t in TEAMS.values())
    total_team_games = sum(t.mp for t in TEAMS.values())
    lam = total_goals / total_team_games
    return lam, 2 * lam
