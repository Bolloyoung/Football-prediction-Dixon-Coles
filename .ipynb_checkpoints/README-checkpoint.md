# Football Predictor — EPL 2025-26

A self-contained Python implementation of the prediction framework
described in the project's theory document.

## Pipeline

```
data.py  ── Season stats (GF, GA, xG, form) + next-matchday fixtures
   │
   ▼
ratings.py ── Elo · Pi-ratings · Berrar-style attack/defense fit from
             blended (goals + xG) season totals
   │
   ▼
model.py ── Dixon-Coles bivariate Poisson with low-score correction
            (ρ = −0.12).  Returns a full scoreline distribution.
   │
   ▼
predict.py ── Runs the pipeline on every fixture in FIXTURES and writes:
               predictions.md      (human-readable report)
               predictions.csv     (1X2 + scoreline probabilities)
               predictions_topN.csv (top-5 scorelines per match)
   │
   ▼
kelly.py ── Kelly stake sizing / positive-EV filter helper to plug in
            your own bookmaker odds.
```

## Run

```bash
cd "Football Prediction."
python -m football_predictor.predict
```

Python 3.9+ is enough.  The only dependency is **numpy**.

## Refreshing the data

Open `data.py`, edit the `_raw` list (one row per team, season totals) and/or
the `FIXTURES` list, and re-run.  The Berrar-style attack/defense fit and
Elo seeds are computed on every run, so a single edit propagates everywhere.

### Form

Each team has a `form` multiplier (1.0 = average).  Bump it toward 1.2
for a hot streak, drop toward 0.7 for a bad run.  The multiplier is
applied to the team's attack rate only.

### xG blend

`XG_BLEND` in `predict.py` (default 0.4) controls how much the
attack/defense fit uses expected goals versus actual goals.  Raising it
damps luck-driven outliers (finishing above/below expectation); lowering
it trusts the scoreboard.

## What the model actually does

For each fixture we compute expected goals

$$\lambda_h = \alpha_h \cdot \beta_a \cdot \mu \cdot \gamma \cdot f_h$$
$$\lambda_a = \alpha_a \cdot \beta_h \cdot \mu / \gamma \cdot f_a$$

where $\alpha$ is attack, $\beta$ is defensive weakness, $\mu$ is the
league per-team goal mean, $\gamma$ is home advantage and $f$ is recent
form.  Goals are distributed Poisson, and the joint probability mass for
a scoreline $(x, y)$ is

$$P(x, y) = \tau(x, y) \cdot \frac{\lambda_h^x e^{-\lambda_h}}{x!}
                           \cdot \frac{\lambda_a^y e^{-\lambda_a}}{y!}$$

with Dixon-Coles low-score correction $\tau$.  We then renormalise the
truncated matrix and derive 1X2, most-likely scoreline, top-N scorelines,
xG, over/under 2.5 and BTTS from that matrix.

Elo is fit from points-per-game and reported as a sanity-check alongside
the Poisson model.

## Kelly filter — example

```python
from football_predictor.kelly import kelly_fraction, expected_value

p = 0.48          # model probability of Home win
odds = 2.30       # decimal odds from bookmaker
print(expected_value(p, odds))       # edge per unit
print(kelly_fraction(p, odds, 0.5))  # half-Kelly stake as % of bankroll
```
