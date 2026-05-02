"""
model.py
--------
Probabilistic scoreline model.

Uses the Dixon-Coles (1997) extension of the bivariate Poisson model:
  * Goals for each team are Poisson with rates lambda_h and lambda_a.
  * The joint distribution is the product of the marginals with a
    correction tau(x, y) for low scorelines (0-0, 1-0, 0-1, 1-1) that
    reflects the observed draw inflation in real football data.

Outputs:
  * Full scoreline matrix P(home=i, away=j) for i, j in [0..max_goals]
  * 1X2 probabilities (home win / draw / away win)
  * Most-likely scoreline, top-N scorelines, expected goals
  * Over/Under 2.5, BTTS probabilities for reference
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# Dixon-Coles low-score correction.  rho in (-1, 1); negative values
# reduce P(1-0) and P(0-1) and lift P(0-0) and P(1-1).  Typical EPL
# calibration is ~ -0.12 (Dixon-Coles 1997, Rue-Salvesen follow-ups).
DEFAULT_RHO = -0.12
MAX_GOALS = 10


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam ** k / math.factorial(k)


def _dc_tau(x: int, y: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Dixon-Coles adjustment factor tau(x,y) for low scorelines."""
    if x == 0 and y == 0:
        return 1 - lam_h * lam_a * rho
    if x == 0 and y == 1:
        return 1 + lam_h * rho
    if x == 1 and y == 0:
        return 1 + lam_a * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


@dataclass
class ScorelineDistribution:
    lam_home: float
    lam_away: float
    matrix: np.ndarray      # P(home=i, away=j)
    max_goals: int

    # ------------------------------------------------------------------
    #  factory
    # ------------------------------------------------------------------

    @classmethod
    def from_rates(cls, lam_home: float, lam_away: float,
                   rho: float = DEFAULT_RHO,
                   max_goals: int = MAX_GOALS) -> "ScorelineDistribution":
        m = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            pi = _poisson_pmf(i, lam_home)
            for j in range(max_goals + 1):
                pj = _poisson_pmf(j, lam_away)
                m[i, j] = pi * pj * _dc_tau(i, j, lam_home, lam_away, rho)
        # Renormalise — the DC correction leaves a tiny imbalance and the
        # truncation at max_goals removes the far tail.
        m = m / m.sum()
        return cls(lam_home=lam_home, lam_away=lam_away,
                   matrix=m, max_goals=max_goals)

    # ------------------------------------------------------------------
    #  derived quantities
    # ------------------------------------------------------------------

    def outcome_probs(self) -> Tuple[float, float, float]:
        """Return (P(home win), P(draw), P(away win))."""
        p_home = float(np.tril(self.matrix, -1).sum())
        p_draw = float(np.trace(self.matrix))
        p_away = float(np.triu(self.matrix, 1).sum())
        return p_home, p_draw, p_away

    def most_likely_scoreline(self) -> Tuple[int, int, float]:
        idx = int(np.argmax(self.matrix))
        h, a = divmod(idx, self.matrix.shape[1])
        return h, a, float(self.matrix[h, a])

    def top_n_scorelines(self, n: int = 5) -> List[Tuple[int, int, float]]:
        flat = self.matrix.flatten()
        order = np.argsort(flat)[::-1][:n]
        out = []
        for idx in order:
            h, a = divmod(int(idx), self.matrix.shape[1])
            out.append((h, a, float(flat[idx])))
        return out

    def expected_goals(self) -> Tuple[float, float]:
        hs = np.arange(self.matrix.shape[0])
        as_ = np.arange(self.matrix.shape[1])
        eh = float((self.matrix.sum(axis=1) * hs).sum())
        ea = float((self.matrix.sum(axis=0) * as_).sum())
        return eh, ea

    def prob_over(self, line: float = 2.5) -> float:
        total = 0.0
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                if i + j > line:
                    total += self.matrix[i, j]
        return float(total)

    def prob_btts(self) -> float:
        """Probability both teams score."""
        return float(self.matrix[1:, 1:].sum())
