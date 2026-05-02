"""
streamlit_app.py
----------------
Interactive dashboard for the football-prediction system.

Tabs
====
1. **Predictions** — view next-matchday H/D/A, scoreline matrix and Kelly
   filter for the live fixtures pulled from fixturedownload.com.
2. **Live editor** — edit team form, injuries, set-piece strength etc.
   on the fly and watch every probability update in real time.
3. **Custom matchup** — pick any two teams (and any kickoff date) and
   blend ML + Dixon-Coles to predict the result.
4. **Refresh data** — trigger a fresh historical-CSV scrape and ML
   retrain from inside the UI.

Run locally:
    streamlit run streamlit_app.py

Deploy:  push to GitHub → streamlit.io/cloud → 'New app' → repo / branch /
         file = streamlit_app.py.  No secrets needed.
"""

from __future__ import annotations

import os
import io
import datetime as dt
from typing import Dict

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

from football_predictor.pipeline import (
    run_full_pipeline, predict_fixture, fit_dixon_coles_from_history,
    acquire_fixtures, acquire_history,
)
from football_predictor.ml_model import (
    build_feature_matrix, train_model, save_bundle, load_bundle, predict_match,
)
from football_predictor.model      import ScorelineDistribution
from football_predictor.kelly      import kelly_fraction, expected_value


# --------------------------------------------------------------------------- #
#  Page config                                                                 #
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Football Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
#  Sidebar — global controls                                                   #
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.title("⚽ Football Predictor")
    st.caption("Dixon-Coles + XGBoost ensemble")
    league = st.selectbox(
        "League", ["EPL", "LaLiga", "Bundesliga", "SerieA", "Ligue1"], index=0,
    )
    season_from = st.number_input("Train from season starting",
                                  2010, 2024, 2018, step=1)
    season_to   = st.number_input("…through season starting",
                                  2010, 2026, 2025, step=1)
    blend = st.slider("ML weight in blend", 0.0, 1.0, 0.5, 0.05,
                      help="0 = pure Dixon-Coles · 1 = pure XGBoost")
    window = st.slider("Fixture window (days)", 1, 30, 14)
    refresh = st.checkbox("Force retrain", value=False)
    model_path = f"models/{league.lower()}.joblib"
    run_btn = st.button("🔄 Run pipeline", type="primary", use_container_width=True)


# --------------------------------------------------------------------------- #
#  Caching                                                                     #
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def _cached_pipeline(league: str, season_from: int, season_to: int,
                     model_path: str, refresh: bool, window: int,
                     blend: float):
    return run_full_pipeline(
        leagues=[league],
        season_from=season_from,
        season_to=season_to,
        model_path=model_path,
        refresh=refresh,
        window_days=window,
        blend_weight=blend,
        verbose=False,
    )


def _ensure_state():
    if "bundle" not in st.session_state:
        st.session_state.bundle = None
    if "edits" not in st.session_state:
        st.session_state.edits = {}


_ensure_state()

if run_btn:
    with st.spinner(f"Scraping {league} history, training XGBoost, "
                    "predicting next fixtures…"):
        try:
            st.session_state.bundle = _cached_pipeline(
                league, int(season_from), int(season_to),
                model_path, refresh, int(window), float(blend),
            )
            st.success("Pipeline complete.")
        except Exception as e:
            st.error(f"Pipeline failed: {e}")


# --------------------------------------------------------------------------- #
#  Tabs                                                                        #
# --------------------------------------------------------------------------- #

tab_pred, tab_edit, tab_custom, tab_refresh = st.tabs([
    "📈 Predictions", "✏️ Live editor",
    "🎯 Custom matchup", "🔁 Refresh data",
])


# ---------- Tab 1: predictions ---------- #
with tab_pred:
    bundle = st.session_state.bundle
    if not bundle:
        st.info("Click **Run pipeline** in the sidebar to load predictions.")
    else:
        preds = bundle["predictions"]
        if not preds:
            st.warning(
                "No upcoming fixtures returned.  This is usually because "
                "fixturedownload.com is unreachable from your environment.  "
                "Use the **Custom matchup** tab to predict any pair manually."
            )
        else:
            df = pd.DataFrame([
                {
                    "Kickoff": p.get("kickoff", ""),
                    "Home": p["home"], "Away": p["away"],
                    "Score": p["score"],
                    "Score %": p["score_prob"] * 100,
                    "H %":  p["p_home"] * 100,
                    "D %":  p["p_draw"] * 100,
                    "A %":  p["p_away"] * 100,
                    "λ home": p["lam_home"], "λ away": p["lam_away"],
                    "O2.5 %": p["over25"] * 100,
                    "BTTS %": p["btts"]  * 100,
                    "Elo H":  p["elo_home"], "Elo A": p["elo_away"],
                }
                for p in preds
            ])
            st.subheader(f"Next-{window}-day {league} fixtures")
            st.dataframe(
                df.style.format({
                    "Score %": "{:.1f}",
                    "H %": "{:.1f}", "D %": "{:.1f}", "A %": "{:.1f}",
                    "O2.5 %": "{:.1f}", "BTTS %": "{:.1f}",
                    "λ home": "{:.2f}", "λ away": "{:.2f}",
                }),
                use_container_width=True, height=320,
            )

            # Heatmap of selected match
            st.markdown("### Scoreline heatmap")
            fxs = [f"{p['home']} v {p['away']}" for p in preds]
            sel = st.selectbox("Match", fxs)
            p = preds[fxs.index(sel)]
            mat = ScorelineDistribution.from_rates(
                p["lam_home"], p["lam_away"]
            ).matrix
            heat_df = pd.DataFrame(mat[:7, :7]).reset_index().melt(
                id_vars="index", var_name="Away", value_name="P"
            ).rename(columns={"index": "Home"})
            chart = alt.Chart(heat_df).mark_rect().encode(
                x="Away:O", y="Home:O",
                color=alt.Color("P:Q", scale=alt.Scale(scheme="blues")),
                tooltip=["Home", "Away", alt.Tooltip("P:Q", format=".3f")],
            ).properties(width=500, height=380)
            text = alt.Chart(heat_df).mark_text(baseline="middle").encode(
                x="Away:O", y="Home:O",
                text=alt.Text("P:Q", format=".2f"),
                color=alt.condition("datum.P > 0.10",
                                    alt.value("white"), alt.value("black")),
            )
            st.altair_chart(chart + text, use_container_width=True)

            # Kelly filter
            st.markdown("### Kelly filter (paste odds in decimal form)")
            odds_h = st.number_input("Odds H", 1.01, 50.0, 2.10, step=0.05)
            odds_d = st.number_input("Odds D", 1.01, 50.0, 3.40, step=0.05)
            odds_a = st.number_input("Odds A", 1.01, 50.0, 3.20, step=0.05)
            rows = []
            for label, prob, odds in [("Home", p["p_home"], odds_h),
                                      ("Draw", p["p_draw"], odds_d),
                                      ("Away", p["p_away"], odds_a)]:
                f = kelly_fraction(prob, odds)
                ev = expected_value(prob, odds)
                rows.append([label, prob, odds, ev, f])
            kdf = pd.DataFrame(rows, columns=["Side", "P", "Odds", "EV", "Kelly"])
            st.dataframe(
                kdf.style.format({"P": "{:.3f}", "Odds": "{:.2f}",
                                  "EV": "{:+.3f}", "Kelly": "{:.3f}"}),
                use_container_width=True,
            )


# ---------- Tab 2: live editor ---------- #
with tab_edit:
    bundle = st.session_state.bundle
    if not bundle:
        st.info("Run the pipeline first.")
    else:
        st.markdown(
            "Tweak attack/defence multipliers and watch the prediction update."
        )
        ad = bundle["ad"]; elo = bundle["elo"]
        teams = sorted(ad.alpha.keys())
        c1, c2 = st.columns(2)
        with c1:
            home = st.selectbox("Home team", teams, key="le_h")
            alpha_h = st.slider("Home attack α",
                                0.3, 2.5, float(ad.alpha[home]), 0.01)
            beta_h  = st.slider("Home defence β",
                                0.3, 2.5, float(ad.beta[home]),  0.01)
            inj_h   = st.slider("Home injury index", 0.0, 1.0, 0.0, 0.05)
        with c2:
            away = st.selectbox("Away team", teams, key="le_a", index=1)
            alpha_a = st.slider("Away attack α",
                                0.3, 2.5, float(ad.alpha[away]), 0.01)
            beta_a  = st.slider("Away defence β",
                                0.3, 2.5, float(ad.beta[away]),  0.01)
            inj_a   = st.slider("Away injury index", 0.0, 1.0, 0.0, 0.05)

        mu = ad.mu; gamma = ad.gamma
        lam_h = alpha_h * beta_a * mu * gamma * (1 - 0.20 * inj_h)
        lam_a = alpha_a * beta_h * mu / gamma * (1 - 0.20 * inj_a)
        dist = ScorelineDistribution.from_rates(lam_h, lam_a)
        p_h, p_d, p_a = dist.outcome_probs()
        mh, ma, pm = dist.most_likely_scoreline()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Home win %", f"{p_h*100:.1f}")
        c2.metric("Draw %",     f"{p_d*100:.1f}")
        c3.metric("Away win %", f"{p_a*100:.1f}")
        c4.metric("Top score",  f"{mh}-{ma}", f"{pm*100:.1f}%")

        st.markdown(f"**λ:** {lam_h:.2f} – {lam_a:.2f}  ·  "
                    f"**O2.5:** {dist.prob_over(2.5)*100:.1f}%  ·  "
                    f"**BTTS:** {dist.prob_btts()*100:.1f}%")


# ---------- Tab 3: custom matchup ---------- #
with tab_custom:
    bundle = st.session_state.bundle
    if not bundle:
        st.info("Run the pipeline first.")
    else:
        ad = bundle["ad"]; elo = bundle["elo"]
        ml_model = bundle["model"]; meta = bundle["meta"]
        teams = sorted(ad.alpha.keys())
        c1, c2, c3 = st.columns(3)
        h = c1.selectbox("Home", teams, key="cu_h")
        a = c2.selectbox("Away", teams, key="cu_a", index=1)
        kickoff = c3.date_input("Kickoff",
                                value=dt.date.today() + dt.timedelta(days=3))
        if h == a:
            st.warning("Pick two different teams.")
        else:
            try:
                p = predict_fixture(h, a, ml_model, meta, ad, elo,
                                    blend_weight=blend, kickoff=kickoff)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Home win %", f"{p['p_home']*100:.1f}",
                          f"ML {p['p_home_ml']*100:.0f}% · DC {p['p_home_dc']*100:.0f}%")
                c2.metric("Draw %",     f"{p['p_draw']*100:.1f}",
                          f"ML {p['p_draw_ml']*100:.0f}% · DC {p['p_draw_dc']*100:.0f}%")
                c3.metric("Away win %", f"{p['p_away']*100:.1f}",
                          f"ML {p['p_away_ml']*100:.0f}% · DC {p['p_away_dc']*100:.0f}%")
                c4.metric("Top score",  p["score"], f"{p['score_prob']*100:.1f}%")

                topn = pd.DataFrame(
                    p["top_scorelines"], columns=["Home", "Away", "P"]
                )
                topn["P"] = topn["P"] * 100
                st.markdown("**Top scorelines**")
                st.dataframe(
                    topn.style.format({"P": "{:.2f}%"}),
                    use_container_width=True, height=240,
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")


# ---------- Tab 4: refresh data ---------- #
with tab_refresh:
    st.markdown(
        "Force a fresh download from football-data.co.uk and retrain "
        "the XGBoost model.  Existing CSVs in `data/raw/` are overwritten."
    )
    if st.button("⚠️ Force fresh scrape + retrain"):
        with st.spinner("Downloading CSVs and retraining…"):
            try:
                st.session_state.bundle = _cached_pipeline(
                    league, int(season_from), int(season_to),
                    model_path, True, int(window), float(blend),
                )
                rep = st.session_state.bundle["model_report"]
                st.success(
                    f"Done.  log-loss={rep.get('test_logloss', float('nan')):.4f}, "
                    f"acc={rep.get('test_accuracy', float('nan')):.3f}, "
                    f"n={rep.get('n_train', 0)}+{rep.get('n_test', 0)}"
                )
            except Exception as e:
                st.error(f"Refresh failed: {e}")

    if st.session_state.bundle:
        st.markdown("### Last training report")
        st.json(st.session_state.bundle["model_report"])

st.caption(
    "Built on Dixon-Coles bivariate Poisson (Maher 1982 · Dixon-Coles 1997) "
    "blended with calibrated XGBoost.  See `EPL_Tutorial.ipynb` for the math."
)
