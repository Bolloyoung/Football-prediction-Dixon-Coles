# Football Predictor — EPL 2025-26 (enhanced)

End-to-end implementation of the framework paper's prediction pipeline with
extra "niche" match-level features and matplotlib visualisations.

## Pipeline

```
data.py ─── Season stats + H2H history + referees + fixtures
   │        (GF, GA, xG, xGA, form, yellows/reds/fouls, set-piece %,
   │         clean-sheet %, injury index, rest days, travel km,
   │         referee, match importance)
   ▼
ratings.py ─ Elo (with margin + home-advantage + K-factor) ·
             Pi-ratings · Berrar-style attack/defence fit from blended
             (goals + xG) season totals
   │
   ▼
features.py ── Context-feature engine that multiplicatively adjusts λ:
               H2H with exponential time-decay ·
               discipline × referee card tendency ·
               rest days · travel fatigue ·
               set-piece efficiency vs opponent frailty ·
               key-player absence (injury_index) ·
               match importance
   │
   ▼
model.py ─── Dixon-Coles bivariate Poisson with low-score correction
             (ρ = −0.12).  Full scoreline matrix.
   │
   ▼
visualize.py ── matplotlib charts (28 PNGs per run)
   │
   ▼
predict.py ── Orchestrator — emits predictions.md, predictions.csv,
              predictions_topN.csv, and a charts/ folder
   │
   ▼
kelly.py ── Positive-EV filter & Kelly stake sizing helper
```

## Run

```bash
cd "Football Prediction."
python -m football_predictor.predict
```

Dependencies: **numpy**, **matplotlib**. Python 3.9+.

## Outputs

| File | Contents |
|---|---|
| `predictions.md` | Human-readable report with summary table, per-match detail, feature contributions. |
| `predictions.csv` | Per-fixture 1X2 probabilities, λ, xG, Elo, H2H. |
| `predictions_topN.csv` | Top-8 scorelines per fixture. |
| `charts/1x2_summary.png` | 1X2 grid for the whole matchday. |
| `charts/scoreline_<H>_vs_<A>.png` | Goals heatmap (home vs away). |
| `charts/features_<H>_vs_<A>.png` | Feature waterfall (H2H, discipline, rest, travel, set-pieces, injuries, importance). |
| `charts/topN_<H>_vs_<A>.png` | Top-8 scoreline bar chart. |
| `charts/attack_defense_scatter.png` | Fitted α vs 1/β for all 20 clubs. |
| `charts/elo_ranking.png` | Elo rankings. |
| `charts/lambda_bars.png` | Expected goals per fixture (home vs away). |

## Refreshing the data

Everything is in `data.py` — edit the `_raw` team rows, the `H2H` list,
the `FIXTURES` list, or the `REFEREES` dict, then rerun. The whole
pipeline refits and redraws in < 2 s.

## Feature adjustments, in one place

| Feature | Source | Effect |
|---|---|---|
| Base λ | Berrar attack × defence × home advantage × form | core rate |
| H2H | time-decayed goal averages (365-day half-life) | ±12 % cap |
| Discipline × Ref | reds/m × ref cards/m | home reds hurt own attack, boost opp |
| Rest days | ≤2 d = 0.90, 3 d = 0.95, 4 d = 0.98, ≥5 d = 1.00 | attack |
| Travel | 250/350/500 km thresholds | away attack |
| Set pieces | attacker's set-piece share × defender's frailty | ±8 % cap |
| Injuries | injury_index → attack × (1 − 0.35·idx), defence × (1 − 0.20·idx) | both sides |
| Importance | imp≥1.3 → ×0.94 on both λ (tighter match) | total goals |

Each feature's contribution is visualised per fixture in `features_*.png`
— the same role SHAP plays in a full XGBoost implementation.

## What the model actually does

For each fixture:

$$\lambda_h = \alpha_h \beta_a \mu \gamma f_h \cdot \prod_k m_h^{(k)}$$
$$\lambda_a = \alpha_a \beta_h (\mu/\gamma) f_a \cdot \prod_k m_a^{(k)}$$

where $\alpha, \beta, \mu, \gamma, f$ come from the ratings layer and
$m^{(k)}$ are the context adjustments from `features.py`.  The joint
scoreline is

$$P(x, y) = \tau(x, y) \cdot \text{Pois}(x; \lambda_h) \cdot \text{Pois}(y; \lambda_a)$$

with Dixon-Coles low-score correction $\tau$.  After truncating at
`MAX_GOALS = 10` and renormalising, we derive 1X2, most-likely
scoreline, top-N, xG, over/under 2.5, BTTS.

## Kelly filter — example

```python
from football_predictor.kelly import kelly_fraction, expected_value

p = 0.48          # model probability of Home win
odds = 2.30       # bookmaker decimal odds
print(expected_value(p, odds))       # edge per unit
print(kelly_fraction(p, odds, 0.5))  # half-Kelly stake as % of bankroll
```

## Extending further

Drop-in extensions that fit the same interface:

* Replace the closed-form Berrar fit with an MLE over full match logs
  (swap `AttackDefense.fit` in `ratings.py`).
* Swap the Poisson model for an XGBoost classifier trained on the feature
  vectors produced by `features.py` and calibrate with isotonic regression
  (the framework's 1X2 approach).
* Add SHAP plots on the XGBoost model alongside the `features_*.png`
  waterfalls — the waterfalls use model-intrinsic factors so both
  explanations live on the same charts grid.
