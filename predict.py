"""
predict.py
----------
End-to-end pipeline following the enhanced framework:

    Data → Ratings → Context Features → Probabilistic Model →
           Calibration → Visualisation → Kelly Filter

Flow:
  1. Fit Berrar attack/defense from season goals blended with xG.
  2. Compute Elo seeds for comparison.
  3. For each fixture:
        a. base λ from attack × defense × home-advantage
        b. apply context-feature multipliers (H2H, discipline, rest,
           travel, set-piece, injuries, importance, referee)
        c. Dixon-Coles bivariate Poisson → full scoreline matrix
        d. derive 1X2, top-N scorelines, xG, O2.5, BTTS
  4. Emit markdown + CSV report.
  5. Render PNG charts (heatmaps, 1X2 grid, attack/defense scatter,
     Elo ranking, λ bars, feature waterfalls, top scorelines).

Run:

    python -m football_predictor.predict
"""

from __future__ import annotations

import csv
import os
import sys
import datetime as dt
from typing import Dict, List

try:
    from .data       import TEAMS, FIXTURES, league_averages, REFEREES
    from .ratings    import AttackDefense, build_elo_from_season
    from .model      import ScorelineDistribution
    from .features   import combine_adjustments, h2h_goal_deltas
    from .visualize  import render_all
except ImportError:  # pragma: no cover
    from data      import TEAMS, FIXTURES, league_averages, REFEREES
    from ratings   import AttackDefense, build_elo_from_season
    from model     import ScorelineDistribution
    from features  import combine_adjustments, h2h_goal_deltas
    from visualize import render_all


# ----------------------------------------------------------------------
#  PATHS / CONFIG
# ----------------------------------------------------------------------

ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORT_MD   = os.path.join(ROOT, "predictions.md")
REPORT_CSV  = os.path.join(ROOT, "predictions.csv")
REPORT_TOPN = os.path.join(ROOT, "predictions_topN.csv")
CHARTS_DIR  = os.path.join(ROOT, "charts")

XG_BLEND = 0.4       # 0=pure goals, 1=pure xG


# ----------------------------------------------------------------------
#  PREDICT ONE MATCH
# ----------------------------------------------------------------------

def predict_one(fx, ad: AttackDefense, elo) -> Dict:
    home, away = fx.home, fx.away

    # -------- base λ from ratings --------
    lam_h0, lam_a0 = ad.expected_goals(
        home, away,
        form_home=TEAMS[home].form,
        form_away=TEAMS[away].form,
    )

    # -------- context-feature adjustments --------
    mult_h, mult_a, parts = combine_adjustments(fx)
    lam_h = lam_h0 * mult_h
    lam_a = lam_a0 * mult_a

    # -------- scoreline distribution --------
    dist = ScorelineDistribution.from_rates(lam_h, lam_a)
    p_h, p_d, p_a = dist.outcome_probs()
    mh, ma, pm = dist.most_likely_scoreline()
    topN = dist.top_n_scorelines(8)
    over25 = dist.prob_over(2.5)
    btts   = dist.prob_btts()
    eh, ea = dist.expected_goals()

    # -------- Elo cross-check --------
    elo_home_win = elo.expectation(home, away)
    draw_share = 0.26
    p_elo_h = (1 - draw_share) * elo_home_win
    p_elo_a = (1 - draw_share) * (1 - elo_home_win)

    # H2H info (for report)
    h2h_h, h2h_a, h2h_n = h2h_goal_deltas(home, away)

    return {
        "kickoff":   fx.kickoff,
        "referee":   fx.referee,
        "home":      home,
        "away":      away,
        "lam_home_base": round(lam_h0, 3),
        "lam_away_base": round(lam_a0, 3),
        "mult_home": round(mult_h, 3),
        "mult_away": round(mult_a, 3),
        "lam_home":  round(lam_h, 3),
        "lam_away":  round(lam_a, 3),
        "xg_home":   round(eh, 3),
        "xg_away":   round(ea, 3),
        "p_home":    round(p_h, 4),
        "p_draw":    round(p_d, 4),
        "p_away":    round(p_a, 4),
        "p_elo_home": round(p_elo_h, 4),
        "p_elo_draw": round(draw_share, 4),
        "p_elo_away": round(p_elo_a, 4),
        "score":     f"{mh}-{ma}",
        "score_prob": round(pm, 4),
        "top_scorelines": topN,
        "over25":    round(over25, 4),
        "btts":      round(btts, 4),
        "elo_home":  round(elo.ratings[home], 1),
        "elo_away":  round(elo.ratings[away], 1),
        "h2h_used":  h2h_n,
        "h2h_home_avg": round(h2h_h, 2),
        "h2h_away_avg": round(h2h_a, 2),
        "matrix":    dist.matrix,
        "breakdown": parts,
    }


# ----------------------------------------------------------------------
#  REPORT
# ----------------------------------------------------------------------

def _bar(p: float, width: int = 20) -> str:
    n = int(round(p * width))
    return "█" * n + "·" * (width - n)


def format_markdown(preds: List[Dict]) -> str:
    out: List[str] = []
    out.append("# EPL 2025-26 — Matchday Predictions (enhanced)\n")
    out.append(
        f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}  •  "
        f"Dixon-Coles bivariate Poisson · Berrar attack/defence "
        f"(xG blend {XG_BLEND:.0%}) · context-feature adjustments "
        f"(H2H, discipline, referee, rest, travel, set-pieces, injuries, importance)\n")
    out.append("\n---\n")

    # summary
    out.append("## Summary\n")
    out.append("| Kickoff | Fixture | Predicted | H % | D % | A % | λ (H–A) | O2.5 | BTTS | Ref |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for p in preds:
        out.append(
            f"| {p['kickoff']} | **{p['home']} v {p['away']}** | "
            f"**{p['score']}** ({p['score_prob']*100:.1f}%) | "
            f"{p['p_home']*100:.1f} | {p['p_draw']*100:.1f} | {p['p_away']*100:.1f} | "
            f"{p['lam_home']:.2f}-{p['lam_away']:.2f} | "
            f"{p['over25']*100:.1f}% | {p['btts']*100:.1f}% | {p['referee']} |")
    out.append("")

    # per-match
    out.append("## Match-by-match detail\n")
    for p in preds:
        out.append(f"### {p['home']} v {p['away']}  — *{p['kickoff']}*")
        out.append("")
        out.append(f"* Base λ: **{p['lam_home_base']:.2f} – {p['lam_away_base']:.2f}**")
        out.append(f"* Adjusted λ: **{p['lam_home']:.2f} – {p['lam_away']:.2f}**  "
                   f"(home ×{p['mult_home']:.2f} · away ×{p['mult_away']:.2f})")
        out.append(f"* Elo: {p['home']} {p['elo_home']:.0f}  vs  "
                   f"{p['away']} {p['elo_away']:.0f}  (ref {p['referee']})")
        if p["h2h_used"] > 0:
            out.append(
                f"* H2H (time-decayed, {p['h2h_used']} recent meetings): "
                f"{p['home']} {p['h2h_home_avg']:.2f} – {p['h2h_away_avg']:.2f} {p['away']}")
        out.append("")
        out.append(f"* **Most-likely scoreline: {p['score']} "
                   f"({p['score_prob']*100:.1f}%)**")
        out.append("")
        out.append("| Outcome | Probability |")
        out.append("|---|---|")
        out.append(f"| {p['home']} win | {p['p_home']*100:5.1f}% `{_bar(p['p_home'])}` |")
        out.append(f"| Draw           | {p['p_draw']*100:5.1f}% `{_bar(p['p_draw'])}` |")
        out.append(f"| {p['away']} win | {p['p_away']*100:5.1f}% `{_bar(p['p_away'])}` |")
        out.append("")
        out.append("Top scorelines:\n")
        out.append("| Score | Probability |")
        out.append("|---|---|")
        for h, a, prob in p["top_scorelines"][:6]:
            out.append(f"| {h}-{a} | {prob*100:.2f}% |")
        out.append("")
        out.append("Feature contributions (×home · ×away):\n")
        out.append("| Feature | Home | Away | Notes |")
        out.append("|---|---|---|---|")
        for b in p["breakdown"]:
            out.append(
                f"| {b.name} | {b.home_factor:.3f} | {b.away_factor:.3f} | {b.note} |")
        out.append("")
        out.append(f"*Over 2.5 goals: {p['over25']*100:.1f}% · BTTS: {p['btts']*100:.1f}%*")
        out.append("\n---\n")

    out.append(
        "### Enhancements vs. the base model\n\n"
        "* **H2H with exponential time decay** (365-day half-life): nudges λ "
        "toward historical matchup patterns, capped at ±12 %.\n"
        "* **Discipline × referee**: each team's red-cards-per-match scaled by "
        "the referee's card tendency; aggressive teams with strict refs lose "
        "attacking output and concede more on transition.\n"
        "* **Rest days / fixture congestion**: <3 days = heavy fatigue penalty.\n"
        "* **Travel fatigue**: long away trips (>350 km, >500 km) shave 1–3 % off "
        "away attack.\n"
        "* **Set-piece efficiency × opponent frailty**: high set-piece scorers "
        "punish opponents with low clean-sheet rates / injured defenders.\n"
        "* **Key-player absence (injury_index)**: drops own attack and concedes "
        "more on opponent's attack.\n"
        "* **Match importance**: high-stakes games (title/relegation) tighten "
        "scoring modestly.\n"
        "* **Visual explanations**: feature waterfalls per fixture show exactly "
        "which factor moved each team's λ — the framework's SHAP role.\n")
    return "\n".join(out)


def write_csv(path: str, preds: List[Dict]) -> None:
    cols = ["kickoff", "referee", "home", "away", "score", "score_prob",
            "p_home", "p_draw", "p_away",
            "lam_home_base", "lam_away_base", "mult_home", "mult_away",
            "lam_home", "lam_away", "xg_home", "xg_away",
            "over25", "btts", "elo_home", "elo_away",
            "h2h_used", "h2h_home_avg", "h2h_away_avg"]
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

    # -------- reports --------
    with open(REPORT_MD, "w") as f:
        f.write(format_markdown(preds))
    write_csv(REPORT_CSV, preds)
    write_topn_csv(REPORT_TOPN, preds)

    # -------- charts --------
    breakdowns = {f"{p['home']}|{p['away']}": p["breakdown"] for p in preds}
    chart_paths = render_all(
        preds, ad.alpha, ad.beta, elo.ratings, breakdowns, CHARTS_DIR
    )

    # -------- terminal summary --------
    print()
    print("Predicted scorelines  (home − away)")
    print("-----------------------------------")
    for p in preds:
        print(f"  {p['home']:<20} {p['score']:<5} "
              f"{p['away']:<20}   "
              f"H/D/A = {p['p_home']*100:4.1f}/"
              f"{p['p_draw']*100:4.1f}/{p['p_away']*100:4.1f}%  "
              f"λ={p['lam_home']:.2f}-{p['lam_away']:.2f}  "
              f"(×{p['mult_home']:.2f}/×{p['mult_away']:.2f})")
    print()
    print(f"Report:    {REPORT_MD}")
    print(f"CSV:       {REPORT_CSV}")
    print(f"Top-N CSV: {REPORT_TOPN}")
    print(f"Charts:    {CHARTS_DIR}  ({len(chart_paths)} PNGs)")


if __name__ == "__main__":
    main()
