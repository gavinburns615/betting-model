# NFL Betting Strategy Backtester

Backtests an NFL spread-betting strategy the right way: strict time-ordering
(nothing a bet uses is computed from data that didn't exist yet), walk-forward
validation (tune on earlier seasons, test on strictly later ones), and a
leak-detection sanity check that has to pass before any result is trusted.

**The honest result: a weak, statistically inconclusive edge (52.0%
out-of-sample win rate, p = 0.14) — not something to bet real money on.**
Getting to that number required catching and fixing a real bug in the first
version of the backtest, which is arguably the more interesting part of this
repo: see [Limitations](#limitations) below.

## Viewing the report

**[Live version](https://claude.ai/code/artifact/e91209a2-8178-494c-be97-81f81b5c44ef)**
— or open [report/presnap_edge_backtest.html](report/presnap_edge_backtest.html)
directly in a browser, no server needed. Either way it shows every pick week
by week — past results checked against final scores, and next week's picks
before kickoff — plus a break-even/"is this specific price worth it" check
on each one.

## How it's built

1. **Data** — [nflverse](https://github.com/nflverse) public schedules via
   `nfl_data_py` (free, no key): final scores, closing spread/total/moneyline
   odds for both sides, and weather, 2016–2026. Zero missing odds values
   across 2,639 completed regular-season games.
2. **Signal** — a per-team "predictability" score built from an existing
   presnap play-prediction model I trained separately
   ([pass-run](https://github.com/gavinburns615/pass-run)), turned into a
   presnap-safe, recency-weighted feature: each team's score going into a
   game only ever uses that team's *prior* games, verified by direct
   recomputation that nothing leaks in from the game being bet on or later.
3. **Strategy** — bet the side a 1-variable logistic regression favors (home
   team's signal minus away team's), direction re-fit from only prior
   seasons at every walk-forward fold — never fixed in advance, never
   peeking at the test season.
4. **Backtest** — expanding-window walk-forward, 2020–2025 test seasons,
   checked against two baselines (always bet home, coin flip) run through
   the identical harness. Full metrics: ROI (flat stake and Kelly-sized),
   Sharpe ratio, max drawdown, win rate by season/month, and a leak-detection
   sanity check that fails loudly if either baseline shows a spurious edge.

## Limitations

The first version of this backtest showed a strong early hot streak and
looked promising. It wasn't: the fitted logistic regression's intercept —
which just encodes each fold's tune-window home-cover base rate, itself
small-sample noise — dominated the coefficient on the actual signal in every
single fold. The predicted probability never crossed 0.5 for the home side,
in any fold, regardless of what the signal said; the "strategy" had quietly
degenerated into a disguised *always bet away* rule. The early hot streak
lined up exactly with a real, temporary, well-documented anomaly (reduced
home-field advantage during 2020's fan-less, COVID-era stadiums), not with
anything the signal was actually measuring.

The fix — betting on the *sign* of the signal difference directly, with the
intercept removed entirely — moved the result from indistinguishable-from-
noise (50.3%, p = 0.80) to a real but weak edge (52.0%, p = 0.14). That's a
genuine improvement from correcting an actual bug, not from re-slicing the
data until something looked good. It's still short of conventional
significance, inconsistent season to season (wins 4 of 6 backtested
seasons), and gets *worse* under Kelly staking, not better, because the
higher assumed win rate makes Kelly size more aggressively into an edge
that's smaller and noisier than that estimate suggests.

Bottom line: this is a rigorous no (or at best inconclusive) result, reported
as one. If a future iteration is going to beat this, it needs a genuinely
different information edge — not a re-tuned threshold on the same signal.

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

## Running the pipeline

```bash
venv/bin/python3 src/fetch_games.py                  # games, closing odds, weather (2016-2026)
venv/bin/python3 src/build_predictability_signal.py  # presnap-safe predictability signal
venv/bin/python3 src/build_dataset.py                 # joins games + signal into one time-indexed dataset
venv/bin/python3 src/backtest_ats.py                  # walk-forward backtest: original (buggy) + corrected strategy + 2 baselines
venv/bin/python3 src/report.py                        # full metrics, sanity check, cumulative P/L chart
venv/bin/python3 src/predict_upcoming.py              # forward-looking picks for the current season
venv/bin/python3 src/top2_backtest.py                 # highest-conviction subset: top 2 picks/week + holdout check
```

`report/presnap_edge_backtest.html` is a static snapshot of the interactive
dashboard built from these outputs; regenerate it by re-running the pipeline
and rebuilding the report (see `src/report.py` for the metrics it's built
from — the dashboard itself is assembled separately from that JSON).

## Repo layout

```
src/          data pipeline, backtest harness, metrics, forward predictions
data/         cached pulls and processed datasets (raw pulls gitignored -- see .gitignore)
output/       backtest results: bet-level parquet, fold summaries, chart
report/       static snapshot of the interactive dashboard
```
