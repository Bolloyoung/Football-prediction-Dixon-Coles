"""
visualize.py
------------
matplotlib charts used to explain and explore the predictions:

  * Scoreline heatmap                     (one PNG per match)
  * 1X2 grid for the whole matchday       (one PNG)
  * Attack vs defense scatter             (one PNG)
  * Elo ranking bar chart                 (one PNG)
  * Expected-goals (lambda) comparison    (one PNG)
  * Feature-adjustment waterfall          (one PNG per match)
  * Top-5 scoreline bars                  (one PNG per match)

Charts are saved as PNG under `<project-root>/charts/`.
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from .data import TEAMS
from .features import AdjustmentBreakdown


# ----------------------------------------------------------------------
#  SHARED STYLE
# ----------------------------------------------------------------------

plt.rcParams.update({
    "figure.autolayout": True,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

HOME_COLOR = "#1b6ca8"
DRAW_COLOR = "#b0b0b0"
AWAY_COLOR = "#c0392b"
CMAP = "viridis"


def _safe(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-")


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# ----------------------------------------------------------------------
#  SCORELINE HEATMAP
# ----------------------------------------------------------------------

def plot_scoreline_heatmap(pred: Dict, out_dir: str,
                           max_goals: int = 6) -> str:
    m = pred["matrix"][:max_goals + 1, :max_goals + 1]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(m * 100.0, cmap=CMAP, origin="lower")
    ax.set_xticks(range(max_goals + 1))
    ax.set_yticks(range(max_goals + 1))
    ax.set_xlabel(f"{pred['away']} goals")
    ax.set_ylabel(f"{pred['home']} goals")
    ax.set_title(f"{pred['home']} vs {pred['away']}  —  scoreline probability (%)")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j] * 100.0
            if v >= 1.0:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="white" if v > 4 else "black", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("P(scoreline) %")
    path = os.path.join(out_dir, f"scoreline_{_safe(pred['home'])}_vs_{_safe(pred['away'])}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
#  1X2 GRID
# ----------------------------------------------------------------------

def plot_1x2_grid(preds: List[Dict], out_dir: str) -> str:
    n = len(preds)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, p in zip(axes, preds):
        probs = [p["p_home"], p["p_draw"], p["p_away"]]
        labels = [p["home"], "Draw", p["away"]]
        colors = [HOME_COLOR, DRAW_COLOR, AWAY_COLOR]
        bars = ax.barh(labels, probs, color=colors)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_title(f"{p['home']} vs {p['away']}  ({p['score']} most likely)")
        for b, v in zip(bars, probs):
            ax.text(v + 0.01, b.get_y() + b.get_height() / 2,
                    f"{v*100:.1f}%", va="center", fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("1X2 probabilities — EPL next matchday", fontsize=14, y=1.02)
    path = os.path.join(out_dir, "1x2_summary.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
#  ATTACK / DEFENSE SCATTER
# ----------------------------------------------------------------------

def plot_attack_defense(alpha: Dict[str, float], beta: Dict[str, float],
                        out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 7))
    for name in alpha:
        x = alpha[name]
        y = 1.0 / beta[name]      # higher = better defence
        ax.scatter(x, y, s=60, alpha=0.85)
        ax.annotate(name, (x, y), xytext=(5, 4), textcoords="offset points",
                    fontsize=9)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.axvline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Attack strength (goals-for vs league mean)")
    ax.set_ylabel("Defence strength (goals-against ratio, higher = better)")
    ax.set_title("Attack vs Defence — fitted Berrar / Maher coefficients")
    path = os.path.join(out_dir, "attack_defense_scatter.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
#  ELO RANKING
# ----------------------------------------------------------------------

def plot_elo_ranking(elo_dict: Dict[str, float], out_dir: str) -> str:
    order = sorted(elo_dict.items(), key=lambda x: x[1])
    names = [n for n, _ in order]
    values = [v for _, v in order]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(names, values, color="#2c3e50")
    ax.axvline(1500, color="red", ls="--", lw=0.8, label="Elo 1500 (avg)")
    ax.set_xlabel("Elo rating")
    ax.set_title("EPL 2025-26 Elo rankings (seeded from points-per-game)")
    ax.legend()
    path = os.path.join(out_dir, "elo_ranking.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
#  LAMBDA BARS (expected goals per fixture)
# ----------------------------------------------------------------------

def plot_lambda_bars(preds: List[Dict], out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{p['home']} v {p['away']}" for p in preds]
    lh = np.array([p["lam_home"] for p in preds])
    la = np.array([p["lam_away"] for p in preds])
    y = np.arange(len(labels))
    ax.barh(y - 0.2, lh, height=0.38, color=HOME_COLOR, label="Home λ")
    ax.barh(y + 0.2, la, height=0.38, color=AWAY_COLOR, label="Away λ")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Expected goals (λ)")
    ax.set_title("Expected goals per fixture (post feature adjustments)")
    ax.legend()
    path = os.path.join(out_dir, "lambda_bars.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
#  FEATURE WATERFALL
# ----------------------------------------------------------------------

def plot_feature_waterfall(pred: Dict, breakdown: List[AdjustmentBreakdown],
                           out_dir: str) -> str:
    """
    Horizontal grouped bars: each feature's multiplicative factor for
    home and away.  A bar at 1.0 is neutral; >1 favours the attack,
    <1 suppresses it.
    """
    names = [b.name for b in breakdown]
    home_f = [b.home_factor for b in breakdown]
    away_f = [b.away_factor for b in breakdown]

    fig, ax = plt.subplots(figsize=(9, 0.55 * len(breakdown) + 2))
    y = np.arange(len(breakdown))
    ax.barh(y - 0.2, [h - 1 for h in home_f], height=0.38,
            color=HOME_COLOR, label=f"{pred['home']} (home)")
    ax.barh(y + 0.2, [a - 1 for a in away_f], height=0.38,
            color=AWAY_COLOR, label=f"{pred['away']} (away)")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Multiplicative adjustment on attack rate (0 = neutral, +0.1 = +10%)")
    ax.set_title(f"Feature contributions — {pred['home']} vs {pred['away']}")
    ax.legend()
    path = os.path.join(out_dir,
        f"features_{_safe(pred['home'])}_vs_{_safe(pred['away'])}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
#  TOP-N SCORELINES
# ----------------------------------------------------------------------

def plot_top_scorelines(pred: Dict, out_dir: str, n: int = 8) -> str:
    tops = pred["top_scorelines"][:n]
    labels = [f"{h}-{a}" for h, a, _ in tops]
    probs = [p for _, _, p in tops]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, probs, color="#2a9d8f")
    ax.set_ylim(0, max(probs) * 1.25)
    ax.set_ylabel("Probability")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title(f"Top {n} scorelines — {pred['home']} vs {pred['away']}")
    for b, v in zip(bars, probs):
        ax.text(b.get_x() + b.get_width() / 2, v + max(probs) * 0.02,
                f"{v*100:.1f}%", ha="center", fontsize=9)
    path = os.path.join(out_dir,
        f"topN_{_safe(pred['home'])}_vs_{_safe(pred['away'])}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
#  MAIN DRIVER
# ----------------------------------------------------------------------

def render_all(preds: List[Dict],
               alpha: Dict[str, float], beta: Dict[str, float],
               elo: Dict[str, float],
               breakdowns: Dict[str, List[AdjustmentBreakdown]],
               out_dir: str) -> List[str]:
    _ensure_dir(out_dir)
    paths: List[str] = []

    paths.append(plot_1x2_grid(preds, out_dir))
    paths.append(plot_attack_defense(alpha, beta, out_dir))
    paths.append(plot_elo_ranking(elo, out_dir))
    paths.append(plot_lambda_bars(preds, out_dir))

    for p in preds:
        paths.append(plot_scoreline_heatmap(p, out_dir))
        paths.append(plot_top_scorelines(p, out_dir))
        key = f"{p['home']}|{p['away']}"
        paths.append(plot_feature_waterfall(p, breakdowns[key], out_dir))

    return paths
