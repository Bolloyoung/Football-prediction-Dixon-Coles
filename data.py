"""
data.py
-------
Season-to-date Premier League 2025-26 team statistics (through 19-Apr-2026)
plus the upcoming matchday fixtures (24-26 April 2026).

This file also hosts the extended feature dictionaries that the enhanced
pipeline consumes:
    - H2H history with time decay
    - Player discipline (cards / match, fouls / match, reds / match)
    - Set-piece and clean-sheet proficiency
    - Rest days, travel distance, match importance, referee assignments
    - Key-player absences

All values are editable.  Everything downstream recomputes in < 1 second.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import datetime as _dt


# ----------------------------------------------------------------------
#  CONSTANTS
# ----------------------------------------------------------------------

DEFAULT_HOME_SHARE_GF = 0.55
DEFAULT_HOME_SHARE_GA = 0.45
HOME_ADVANTAGE        = 1.30      # multiplicative on home attack
REFEREE_BASE_CARDS    = 3.60      # league average cards / match
LEAGUE_DRAW_RATE      = 0.26      # baseline draw rate

TODAY = _dt.date(2026, 4, 20)


# ----------------------------------------------------------------------
#  TEAM SEASON RECORD
# ----------------------------------------------------------------------

@dataclass
class TeamSeason:
    name: str
    mp: int
    w: int
    d: int
    l: int
    gf: int
    ga: int
    xg: float
    xga: float

    # Home / away split overrides
    gf_home: float = None
    ga_home: float = None
    gf_away: float = None
    ga_away: float = None

    # Recent form (0.5 = cold, 1.0 = avg, 1.5 = hot).
    form: float = 1.0

    # -- Discipline & style --
    yellows_pg: float = 1.8          # yellow cards / match
    reds_pg:    float = 0.08         # red cards / match
    fouls_pg:   float = 11.0
    # Fraction of goals scored from set-pieces (corners + free-kicks + pens)
    set_piece_share:    float = 0.28
    # Clean sheet rate (GA = 0 fraction of matches)
    clean_sheet_rate:   float = 0.25

    # -- Personnel --
    # Sum of impact ratings (0..1 each) for absent first-team regulars.
    # 0.0 = full-strength; 0.3 typical; >0.6 severely weakened.
    injury_index: float = 0.05

    def __post_init__(self):
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
# (name, MP, W, D, L, GF, GA, xG, xGA, form,
#  yellows_pg, reds_pg, fouls_pg, setpiece%, clean_sheet%, injury)

_raw: List[Tuple] = [
    ("Arsenal",         31, 21, 7, 3,  63, 26, 58.0, 28.3, 1.15, 1.6, 0.04, 10.0, 0.32, 0.42, 0.05),
    ("Manchester City", 30, 19, 4, 7,  65, 33, 56.5, 34.5, 1.25, 1.5, 0.04,  9.8, 0.24, 0.33, 0.10),
    ("Manchester United",31,16, 7, 8,  54, 41, 48.0, 44.0, 1.00, 1.9, 0.08, 11.2, 0.30, 0.26, 0.15),
    ("Aston Villa",     31, 15, 9, 7,  55, 50, 52.0, 48.0, 1.00, 2.1, 0.08, 11.8, 0.27, 0.19, 0.05),
    ("Liverpool",       31, 13,10, 8,  58, 50, 60.0, 46.0, 1.10, 1.7, 0.06, 10.5, 0.25, 0.23, 0.08),
    ("Chelsea",         31, 13, 9, 9,  53, 38, 61.7, 42.0, 1.05, 1.9, 0.10, 11.4, 0.26, 0.29, 0.10),
    ("Brentford",       31, 12,10, 9,  50, 46, 47.0, 46.5, 0.95, 1.8, 0.08, 10.9, 0.40, 0.19, 0.05),
    ("Everton",         31, 12,10, 9,  44, 42, 41.0, 43.0, 1.00, 2.3, 0.12, 12.1, 0.34, 0.29, 0.10),
    ("Fulham",          31, 11,11, 9,  45, 46, 44.0, 47.0, 1.00, 1.9, 0.06, 11.0, 0.31, 0.19, 0.08),
    ("Brighton",        31, 11,10,10,  49, 45, 52.0, 46.0, 0.95, 1.6, 0.04, 10.2, 0.24, 0.23, 0.05),
    ("Sunderland",      31, 11,10,10,  42, 46, 40.0, 48.0, 1.00, 2.2, 0.10, 11.6, 0.33, 0.23, 0.10),
    ("Newcastle",       31, 11, 9,11,  47, 48, 48.0, 45.0, 1.05, 2.0, 0.06, 11.3, 0.35, 0.23, 0.15),
    ("Bournemouth",     31, 11, 9,11,  46, 48, 44.0, 49.0, 0.90, 2.1, 0.08, 11.7, 0.28, 0.16, 0.10),
    ("Crystal Palace",  30, 10, 9,11,  33, 35, 46.3, 40.0, 0.95, 2.2, 0.08, 12.0, 0.37, 0.27, 0.05),
    ("Leeds",           31,  8, 9,14,  38, 49, 37.0, 48.0, 1.20, 2.4, 0.10, 12.5, 0.32, 0.16, 0.10),
    ("Nottingham Forest",31, 8, 8,15,  36, 48, 35.0, 49.5, 0.85, 2.4, 0.12, 12.6, 0.36, 0.19, 0.20),
    ("West Ham",        32,  8, 8,16,  35, 52, 36.0, 50.0, 0.85, 2.3, 0.10, 12.2, 0.34, 0.16, 0.15),
    ("Tottenham",       31,  8, 6,17,  41, 51, 45.0, 48.0, 0.80, 2.0, 0.10, 11.5, 0.29, 0.13, 0.25),
    ("Burnley",         31,  5, 5,21,  26, 54, 28.0, 58.0, 0.70, 2.5, 0.12, 13.0, 0.38, 0.10, 0.20),
    ("Wolves",          30,  3, 6,21,  22, 55, 26.0, 57.0, 0.60, 2.6, 0.14, 13.2, 0.41, 0.10, 0.30),
]

TEAMS: Dict[str, TeamSeason] = {
    r[0]: TeamSeason(
        name=r[0], mp=r[1], w=r[2], d=r[3], l=r[4],
        gf=r[5], ga=r[6], xg=r[7], xga=r[8], form=r[9],
        yellows_pg=r[10], reds_pg=r[11], fouls_pg=r[12],
        set_piece_share=r[13], clean_sheet_rate=r[14], injury_index=r[15],
    )
    for r in _raw
}


# ----------------------------------------------------------------------
#  HEAD-TO-HEAD HISTORY  (recent meetings; most recent last)
# ----------------------------------------------------------------------
#
# Each row: (date, home, away, home_goals, away_goals)
#
# This is a curated sample of each fixture's last-few meetings; the feature
# engine applies exponential time-decay so only ~ the last 2-3 years carry
# meaningful weight.

H2H: List[Tuple[_dt.date, str, str, int, int]] = [
    # Sunderland vs Nottingham Forest
    (_dt.date(2025,11, 8), "Nottingham Forest","Sunderland",       1, 1),
    (_dt.date(2017, 8,26), "Nottingham Forest","Sunderland",       3, 1),
    (_dt.date(2017, 4, 4), "Sunderland",        "Nottingham Forest",1, 0),
    # Fulham vs Aston Villa
    (_dt.date(2025,12,14), "Aston Villa",       "Fulham",          2, 1),
    (_dt.date(2024,11,23), "Fulham",            "Aston Villa",     1, 3),
    (_dt.date(2024, 4,20), "Aston Villa",       "Fulham",          3, 3),
    # Liverpool vs Crystal Palace
    (_dt.date(2025,10, 5), "Crystal Palace",    "Liverpool",       2, 1),
    (_dt.date(2024,12,14), "Liverpool",         "Crystal Palace",  1, 1),
    (_dt.date(2024, 4,14), "Liverpool",         "Crystal Palace",  0, 1),
    # West Ham vs Everton
    (_dt.date(2026, 1,18), "Everton",           "West Ham",        1, 1),
    (_dt.date(2024,11, 9), "West Ham",          "Everton",         0, 1),
    (_dt.date(2024, 5,18), "Everton",           "West Ham",        1, 1),
    # Wolves vs Tottenham
    (_dt.date(2025, 9,21), "Tottenham",         "Wolves",          1, 1),
    (_dt.date(2024,11, 9), "Wolves",            "Tottenham",       2, 2),
    (_dt.date(2024, 2,17), "Tottenham",         "Wolves",          2, 1),
    # Arsenal vs Newcastle
    (_dt.date(2025, 9,28), "Newcastle",         "Arsenal",         2, 1),
    (_dt.date(2024,11, 2), "Arsenal",           "Newcastle",       1, 0),
    (_dt.date(2024, 5,11), "Newcastle",         "Arsenal",         0, 2),
    # Burnley vs Manchester City
    (_dt.date(2024, 2,20), "Manchester City",   "Burnley",         3, 1),
    (_dt.date(2023, 8,11), "Burnley",           "Manchester City", 0, 3),
    (_dt.date(2023, 4,22), "Manchester City",   "Burnley",         6, 0),
    # Brighton vs Chelsea
    (_dt.date(2025,12, 6), "Chelsea",           "Brighton",        1, 3),
    (_dt.date(2024, 9,28), "Brighton",          "Chelsea",         4, 2),
    (_dt.date(2024, 5,15), "Chelsea",           "Brighton",        2, 1),
]


# ----------------------------------------------------------------------
#  REFEREES
# ----------------------------------------------------------------------
#
# Avg cards issued per match for the EPL 2025-26 season.

REFEREES: Dict[str, float] = {
    "Michael Oliver":     3.10,
    "Anthony Taylor":     4.45,
    "Simon Hooper":       4.15,
    "Andrew Madley":      3.40,
    "Craig Pawson":       3.90,
    "Jarred Gillett":     3.80,
    "Stuart Attwell":     4.20,
    "Chris Kavanagh":     4.00,
    "Unknown":            REFEREE_BASE_CARDS,
}


# ----------------------------------------------------------------------
#  FIXTURES  (24-26 Apr 2026)
# ----------------------------------------------------------------------

@dataclass
class Fixture:
    home: str
    away: str
    kickoff:       str
    date:          _dt.date
    rest_home:     int = 7       # days since last match
    rest_away:     int = 7
    travel_km:     int = 200     # away-team travel
    referee:       str = "Unknown"
    importance:    float = 1.0   # 0.8=dead rubber, 1.0=std, 1.2=relegation, 1.3=title


FIXTURES: List[Fixture] = [
    Fixture("Sunderland",        "Nottingham Forest", "Fri 24 Apr 20:00 BST",
            _dt.date(2026,4,24), rest_home=7, rest_away=7,  travel_km=250,
            referee="Andrew Madley",   importance=1.10),
    Fixture("Fulham",             "Aston Villa",       "Sat 25 Apr 12:30 BST",
            _dt.date(2026,4,25), rest_home=7, rest_away=4,  travel_km=170,
            referee="Jarred Gillett",  importance=1.00),
    Fixture("Liverpool",          "Crystal Palace",    "Sat 25 Apr 15:00 BST",
            _dt.date(2026,4,25), rest_home=6, rest_away=7,  travel_km=320,
            referee="Simon Hooper",    importance=1.20),
    Fixture("West Ham",           "Everton",           "Sat 25 Apr 15:00 BST",
            _dt.date(2026,4,25), rest_home=7, rest_away=7,  travel_km=340,
            referee="Craig Pawson",    importance=1.15),
    Fixture("Wolves",             "Tottenham",         "Sat 25 Apr 15:00 BST",
            _dt.date(2026,4,25), rest_home=7, rest_away=7,  travel_km=210,
            referee="Stuart Attwell",  importance=1.25),
    Fixture("Arsenal",            "Newcastle",         "Sat 25 Apr 17:30 BST",
            _dt.date(2026,4,25), rest_home=6, rest_away=3,  travel_km=440,
            referee="Michael Oliver",  importance=1.25),
    Fixture("Burnley",            "Manchester City",   "Sun 26 Apr 14:00 BST",
            _dt.date(2026,4,26), rest_home=7, rest_away=4,  travel_km= 60,
            referee="Anthony Taylor",  importance=1.25),
    Fixture("Brighton",           "Chelsea",           "Sun 26 Apr 16:30 BST",
            _dt.date(2026,4,26), rest_home=7, rest_away=4,  travel_km= 90,
            referee="Chris Kavanagh",  importance=1.10),
]


def league_averages() -> Tuple[float, float]:
    total_goals = sum(t.gf for t in TEAMS.values())
    total_team_games = sum(t.mp for t in TEAMS.values())
    lam = total_goals / total_team_games
    return lam, 2 * lam
