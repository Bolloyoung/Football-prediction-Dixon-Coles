"""
predict.py
----------
End-to-end pipeline that produces score and 1X2 predictions for the next
scheduled EPL matchday.

Flow (matches the framework paper):

    Data  →  Features (ratings + form)  →  Model (Dixon-Coles bivariate
    Poisson)  →  Probabilities  →  Report

Run:

    python -m football_predictor.predict
    # or from this folder directly:
    python predict.py
"""

from __future__ import annotations

import csv
import os
import sys
import datetime as dt
from typing import List, Dict

# Allow running as a script OR as a module.
try:
    from .data import TEAMS, FIXTURES, league_averages
    from .ratings import AttackDefense, build_elo_from_season
    from .model import ScorelineDistribution
except ImportError:  # pragma: no cover
    from data import TEAMS, FIXTURES, league_averages
    from ratings import AttackDefense, build_elo_from_season
    from model import ScorelineDistribution


# ----------------------------------------------------------------------
#  CONFIG
# ----------------------------------------------------------------------

REPORT_MD   = os.path.join(os.path.dirname(__file__), "..", "predictions.md")
REPORT_CSV  = os.path.join(os.path.dirname(__file__), "..", "predictions.csv")
REPORT_TOPN = os.path.join(os.path.dirname(__file__), "..", "predictions_topN.csv")

# How much xG to blend into the attack/defense fit (0 = pure goals, 1 = pure xG).
XG_BLEND = 0.4


# ----------------------------------------------------------------------
#  PREDICT ONE MATCH
# ----------------------------------------------------------------------

def predict_one(fixture, ad: AttackDefense, elo) -> Dict:
    home = fixture.home
    away = fixture.away

    # (1) expected goals from attack / defense + form
    lam_h, lam_a = ad.expected_goals(
        home, away,
        form_home=TEAMS[home].form,
        form_away=TEAMS[away].form,
    )

    # (2) full scoreline distribution with Dixon-Coles correction
    dist = ScorelineDistribution.from_rates(lam_h, lam_a)
    p_h, p_d, p_a = dist.outcome_probs()
    mh, ma, pm = dist.most_likely_scoreline()
    topN = dist.top_n_scorelines(5)
    over25 = dist.prob_over(2.5)
    btts   = dist.prob_btts()
    eh, ea = dist.expected_goals()

    # (3) secondary Elo-based 1X2 probabilities, for comparison
    elo_home_win = elo.expectation(home, away)
    # convert Elo two-way expectation into 1X2 with a typical 26 % draw rate
    draw_share = 0.26
    p_elo_h = (1 - draw_share) * elo_home_win
    p_elo_a = (1 - draw_share) * (1 - elo_home_win)
    p_elo_d = draw_share

    return {
        "kickoff":   fixture.kickoff,
        "home":      home,
        "away":      away,
        "lam_home":  round(lam_h, 3),
        "lam_away":  round(lam_a, 3),
        "xg_home":   round(eh, 3),
        "xg_away":   round(ea, 3),
        "p_home":    round(p_h, 4),
        "p_draw":    round(p_d, 4),
        "p_away":    round(p_a, 4),
        "p_elo_home": round(p_elo_h, 4),
        "p_elo_draw": round(p_elo_d, 4),
        "p_elo_away": round(p_elo_a, 4),
        "score":     f"{mh}-{ma}",
        "score_prob": round(pm, 4),
        "top_scorelines": topN,
        "over25":    round(over25, 4),
        "btts":      round(btts, 4),
        "elo_home":  round(elo.ratings[home], 1),
        "elo_away":  round(elo.ratings[away], 1),
    }


# ----------------------------------------------------------------------
#  REPORT
# ----------------------------------------------------------------------

def _bar(p: float, width: int = 20) -> str:
    n = int(round(p * width))
    return "█" * n + "·" * (width - n)


def format_markdown(preds: List[Dict]) -> str:
    lines: List[str] = []
    lines.append("# EPL 2025-26 — Matchday Predictions")
    lines.append(
        f"\nGenerated: {dt.datetime.now():%Y-%m-%d %H:%M} "
        f"• Model: Dixon-Coles bivariate Poisson with Berrar attack/defense "
        f"(xG blend = {XG_BLEND:.0%}) and Elo comparison.")
    lines.append("\n---\n")

    # summary table
    lines.append("## Summary\n")
    lines.append("| Kickoff | Fixture | Predicted score | Home % | Draw % | Away % | xG (H-A) | O2.5 | BTTS |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for p in preds:
        lines.append(
            f"| {p['kickoff']} | **{p['home']} vs {p['away']}** | "
            f"**{p['score']}** ({p['score_prob']*100:.1f}%) | "
            f"{p['p_home']*100:.1f} | {p['p_draw']*100:.1f} | {p['p_away']*100:.1f} | "
            f"{p['xg_home']:.2f}-{p['xg_away']:.2f} | "
            f"{p['over25']*100:.1f}% | {p['btts']*100:.1f}% |"
        )
    lines.append("")

    # per-match detail
    lines.append("## Match-by-match detail\n")
    for p in preds:
        lines.append(f"### {p['home']} vs {p['away']}  — *{p['kickoff']}*")
        lines.append("")
        lines.append(f"* Model λ (expected goals): **{p['home']} {p['lam_home']:.2f} – "
                     f"{p['lam_away']:.2f} {p['away']}**")
        lines.append(f"* Elo:  {p['home']} {p['elo_home']:.0f}  vs  "
                     f"{p['away']} {p['elo_away']:.0f}")
        lines.append("")
        lines.append(f"* **Most-likely scoreline:** {p['score']} "
                     f"(p = {p['score_prob']*100:.1f}%)")
        lines.append("")
        lines.append("| Outcome | Probability |")
        lines.append("|---|---|")
        lines.append(f"| {p['home']} win | {p['p_home']*100:5.1f}% "
                     f"`{_bar(p['p_home'])}` |")
        lines.append(f"| Draw           | {p['p_draw']*100:5.1f}% "
                     f"`{_bar(p['p_draw'])}` |")
        lines.append(f"| {p['away']} win | {p['p_away']*100:5.1f}% "
                     f"`{_bar(p['p_away'])}` |")
        lines.append("")
        lines.append("*Top 5 most-likely scorelines:*\n")
        lines.append("| Score | Probability |")
        lines.append("|---|---|")
        for h, a, prob in p['top_scorelines']:
            lines.append(f"| {h}-{a} | {prob*100:.2f}% |")
        lines.append("")
        lines.append(f"*Over 2.5 goals: {p['over25']*100:.1f}% · "
                     f"BTTS: {p['btts']*100:.1f}%*")
        lines.append("\n---\n")

    lines.append(
        "### Notes\n\n"
        "* Probabilities come from the Dixon-Coles bivariate Poisson with a "
        "low-score correction (ρ = −0.12) to match the observed draw "
        "inflation in football data.\n"
        "* Attack/defense strengths are fit from season-to-date goals blended "
        "with xG (xG weight "
        f"{XG_BLEND:.0%}) and adjusted multiplicatively by recent-form ratings.\n"
        "* Home advantage is applied as a 1.30× multiplier on the home team's "
        "attack and 1/1.30 on the away team's attack.\n"
        "* Edit `data.py` and re-run to refresh — the whole pipeline is < 1 s.\n"
        "* Use `kelly.py` with your own bookmaker odds to size any stakes "
        "you decide to take.")
    return "\n".join(lines)


def write_csv(path: str, preds: List[Dict]) -> None:
    cols = ["kickoff", "home", "away", "score", "score_prob",
            "p_home", "p_draw", "p_away",
            "p_elo_home", "p_elo_draw", "p_elo_away",
            "lam_home", "lam_away", "xg_home", "xg_away",
            "over25", "btts", "elo_home", "elo_away"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for p in preds:
            w.writerow([p[c] for c in cols])


def write_topn_csv(path: str, preds: List[Dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["home", "away", "rank", "home_goals", "away_goals", "probability"])
        for p in preds:
            for r, (hg, ag, pr) in enumerate(p["top_scorelines"], 1):
                w.writerow([p["home"], p["away"], r, hg, ag, round(pr, 6)])


# ----------------------------------------------------------------------
#  MAIN
# ----------------------------------------------------------------------

def main() -> None:
    mu_team, mu_match = league_averages()
    print(f"[info] League averages: {mu_team:.3f} goals/team-game "
          f"({mu_match:.3f} per match).")

    ad  = AttackDefense.fit(TEAMS, xg_weight=XG_BLEND)
    elo = build_elo_from_season()

    print("[info] Fitted attack/defense for", len(ad.alpha), "teams.")
    preds = [predict_one(fx, ad, elo) for fx in FIXTURES]

    # write outputs
    md_path = os.path.abspath(REPORT_MD)
    csv_path = os.path.abspath(REPORT_CSV)
    top_path = os.path.abspath(REPORT_TOPN)

    with open(md_path, "w") as f:
        f.write(format_markdown(preds))
    write_csv(csv_path, preds)
    write_topn_csv(top_path, preds)

    # quick terminal summary
    print()
    print("Predicted scorelines  (home − away)")
    print("-----------------------------------")
    for p in preds:
        print(f"  {p['home']:<20} {p['score']:<5} "
              f"{p['away']:<20}   "
              f"H/D/A = {p['p_home']*100:4.1f}/{p['p_draw']*100:4.1f}/"
              f"{p['p_away']*100:4.1f}%")
    print()
    print(f"Report:    {md_path}")
    print(f"CSV:       {csv_path}")
    print(f"Top-N CSV: {top_path}")


if __name__ == "__main__":
    main()
