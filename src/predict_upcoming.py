"""Forward-looking predictions for the current/upcoming NFL season, using the
corrected relative-sign strategy (see backtest_ats.py: bet on the sign of
signal_diff = home predictability - away predictability, direction fixed by
the sign found across ALL backtested seasons).

One deliberate difference from the backtest: the backtest's own entering-value
resets to NaN at each season boundary (matching pass-run's build_features.py
convention, which treats "current form" as not carrying across an offseason).
That convention means week 1 of any season has no signal at all -- fine for a
historical test, useless for a live system, since it would never have a week-1
pick. Here, each team's predictability EWMA carries continuously across the
season boundary instead, taking its value as of that team's most recent
completed game -- the same "_through" convention pass-run's own repo already
uses for ITS live predictions (team_state_latest.parquet etc.). This script's
picks are therefore not literally the backtested strategy replayed forward;
they're the same rule with the more realistic live-carry convention.
"""
import json

import pandas as pd

from backtest_ats import fit_direction, load_bettable_games
from build_predictability_signal import RECENCY_SPAN, per_game_scores

PREDICTIONS_PATH = "/Users/gavinburns/pass-run/data/backtest_predictions.parquet"
GAMES_PATH = "data/raw/games_odds.parquet"
OUTPUT_PATH = "data/processed/upcoming_predictions.json"

TEAM_CODE_FIX = {"OAK": "LV", "SD": "LAC"}


def current_team_state() -> pd.Series:
    """Each team's predictability EWMA as of its most recently completed game,
    carrying continuously across season boundaries (no reset)."""
    preds = pd.read_parquet(PREDICTIONS_PATH)
    scores = per_game_scores(preds).sort_values(["posteam", "season", "week"])
    ewma = scores.groupby("posteam")["model_confidence"].transform(
        lambda s: s.ewm(span=RECENCY_SPAN, adjust=False).mean()
    )
    scores = scores.assign(model_confidence_ewma=ewma)
    latest = scores.sort_values(["posteam", "season", "week"]).groupby("posteam").tail(1)
    return latest.set_index("posteam")["model_confidence_ewma"]


def main():
    current_state = current_team_state()

    tune_df = load_bettable_games()  # all 2018-2025 backtested games -- the full history
    _, stats = fit_direction(tune_df)
    coef_sign = 1.0 if stats["coef"] >= 0 else -1.0
    print(f"Direction from full 2018-2025 history: coef={stats['coef']:+.3f} -> "
          f"betting {'HOME' if coef_sign > 0 else 'AWAY'} when home is relatively more predictable")

    games = pd.read_parquet(GAMES_PATH)
    upcoming = games[(games["season"] == 2026) & (games["game_type"] == "REG")].copy()
    upcoming["home_team_join"] = upcoming["home_team"].replace(TEAM_CODE_FIX)
    upcoming["away_team_join"] = upcoming["away_team"].replace(TEAM_CODE_FIX)
    upcoming["gameday"] = pd.to_datetime(upcoming["gameday"]).dt.strftime("%Y-%m-%d")

    records = []
    missing_teams = set()
    for _, g in upcoming.iterrows():
        home, away = g["home_team_join"], g["away_team_join"]
        if home not in current_state.index or away not in current_state.index:
            missing_teams.update({home, away} - set(current_state.index))
            continue

        home_sig = float(current_state[home])
        away_sig = float(current_state[away])
        signal_diff = home_sig - away_sig
        if signal_diff == 0:
            bet_side, no_pick = "home", True
        else:
            bet_side = "home" if (signal_diff * coef_sign) > 0 else "away"
            no_pick = False

        has_odds = pd.notna(g["spread_line"]) and pd.notna(g["home_spread_odds"]) and pd.notna(g["away_spread_odds"])
        bet_odds = None
        if has_odds:
            bet_odds = float(g["home_spread_odds"] if bet_side == "home" else g["away_spread_odds"])

        records.append({
            "season": 2026, "week": int(g["week"]), "gameday": g["gameday"], "game_id": g["game_id"],
            "home_team": g["home_team"], "away_team": g["away_team"],
            "spread_line": float(g["spread_line"]) if pd.notna(g["spread_line"]) else None,
            "has_odds": bool(has_odds), "bet_side": bet_side, "bet_odds": bet_odds, "no_pick": no_pick,
            "home_signal": round(home_sig, 4), "away_signal": round(away_sig, 4),
        })

    if missing_teams:
        print(f"WARNING: no current predictability state for: {missing_teams} (bye/relocation edge case)")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} upcoming-game predictions to {OUTPUT_PATH}")

    n_with_odds = sum(r["has_odds"] for r in records)
    print(f"{n_with_odds} of {len(records)} 2026 games have market lines posted so far")
    print("\nWeek 1 predictions:")
    for r in records:
        if r["week"] == 1:
            pick_team = r["home_team"] if r["bet_side"] == "home" else r["away_team"]
            print(f"  {r['away_team']} @ {r['home_team']}: pick {pick_team} "
                  f"(line {r['spread_line']}, odds {r['bet_odds']})" if r["has_odds"]
                  else f"  {r['away_team']} @ {r['home_team']}: pick {pick_team} (line TBD)")


if __name__ == "__main__":
    main()
