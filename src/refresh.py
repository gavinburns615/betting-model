"""One command to refresh everything the report shows:

    venv/bin/python3 src/refresh.py

Re-pulls schedules/scores/lines, rebuilds the joined dataset, regenerates the
current-season picks (grading any games that have been played and flagging
lines that moved), then rebuilds report/presnap_edge_backtest.html from
report/template.html. Historical (2020-2025) results don't change between
refreshes; they're rebuilt from the saved backtest so the report is fully
reproducible from code.

What a refresh does NOT do: change which side a pick is on. The side comes from
the signal alone (see predict_upcoming.py); prices, the worth-it check, line-move
flags and graded results are what update. It also doesn't fold newly played 2026
games into the signal itself -- that lives in the separate pass-run repo's
play-by-play and would need to be re-run there.
"""
import datetime
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from backtest_ats import SIGNAL_COL, fit_direction, load_bettable_games, tune_win_rate_for_rule
from odds_utils import american_to_implied_prob

BETS_PATH = "data/processed/backtest_bets.parquet"
UPCOMING_PATH = "data/processed/upcoming_predictions.json"
GAMES_LOG_PATH = "output/games_log.json"
COMBINED_PATH = "output/combined_games_log.json"
CURVE_PATH = "output/top2_cumulative_curve.json"
TEMPLATE_PATH = "report/template.html"
REPORT_PATH = "report/presnap_edge_backtest.html"

HISTORICAL_COLS = [
    "season", "week", "gameday", "game_id", "home_team", "away_team", "home_score", "away_score",
    "spread_line", "bet_side", "bet_odds", "bet_won", "bet_prob", "implied_prob", "edge", "worth_it", "is_top2",
]


def run_step(script: str) -> None:
    print(f"\n=== {script} ===")
    subprocess.run([sys.executable, f"src/{script}"], check=True)


def build_historical_log() -> list[dict]:
    rel = pd.read_parquet(BETS_PATH)
    rel = rel[rel["strategy"] == "predictability_signal_relative"].copy()
    rel["abs_signal"] = rel[SIGNAL_COL].abs()
    rel["rank_in_week"] = rel.groupby(["season", "week"])["abs_signal"].rank(ascending=False, method="first")
    rel["is_top2"] = rel["rank_in_week"] <= 2
    rel["gameday"] = pd.to_datetime(rel["gameday"]).dt.strftime("%Y-%m-%d")
    implied = rel["bet_odds"].apply(american_to_implied_prob)
    rel["implied_prob"] = implied.round(4)
    rel["edge"] = (rel["bet_prob"] - implied).round(4)
    rel["worth_it"] = rel["edge"] > 0
    out = rel[HISTORICAL_COLS].sort_values(["season", "week", "gameday"]).to_dict("records")
    for r in out:
        r["future"] = False
    return out


def build_current_log(win_rate: float) -> list[dict]:
    with open(UPCOMING_PATH) as f:
        upcoming = json.load(f)
    for r in upcoming:
        r["bet_prob"] = win_rate
        if r["has_odds"]:
            implied = american_to_implied_prob(r["bet_odds"])
            r["implied_prob"] = round(implied, 4)
            r["edge"] = round(win_rate - implied, 4)
            r["worth_it"] = r["edge"] > 0
        else:
            r["implied_prob"] = r["edge"] = r["worth_it"] = None
        r["future"] = True
    return upcoming


def main():
    for script in ["fetch_games.py", "build_dataset.py", "predict_upcoming.py"]:
        run_step(script)

    # Win-rate estimate behind the worth-it check: the corrected rule's win rate over
    # all backtested history (same number the live picks were always built on).
    history = load_bettable_games()
    _, stats = fit_direction(history)
    win_rate = tune_win_rate_for_rule(history, 1.0 if stats["coef"] >= 0 else -1.0)

    combined = build_historical_log() + build_current_log(win_rate)
    with open(GAMES_LOG_PATH, "w") as f:
        json.dump([r for r in combined if not r["future"]], f)
    with open(COMBINED_PATH, "w") as f:
        json.dump(combined, f)

    with open(CURVE_PATH) as f:
        curve = json.load(f)

    today = datetime.date.today().strftime("%b %-d, %Y")
    with open(TEMPLATE_PATH) as f:
        html = f.read()
    html = (html.replace("__GAMES_LOG_PAYLOAD__", json.dumps(combined))
                .replace("__TOP2_CURVE_PAYLOAD__", json.dumps(curve))
                .replace("__DATA_ASOF__", today))
    with open(REPORT_PATH, "w") as f:
        f.write(html)

    cur = [r for r in combined if r["future"]]
    graded = [r for r in cur if r["completed"] and r["bet_won"] is not None]
    moved = [r for r in cur if r["line_moved_significantly"]]
    print(f"\nRefreshed {today}: {len(cur)} current-season games, {sum(r['has_odds'] for r in cur)} priced, "
          f"{len(graded)} graded ({sum(r['bet_won'] for r in graded)}-{sum(not r['bet_won'] for r in graded)}), "
          f"{len(moved)} with a flagged line move.")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
