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
FIRST_SEEN_PATH = "data/processed/first_seen_spreads.json"

TEAM_CODE_FIX = {"OAK": "LV", "SD": "LAC"}
LINE_MOVE_FLAG_THRESHOLD = 1.0  # points; the pick's SIDE never changes with the line
# (see module docstring / README) -- this only flags that the market has moved since
# the pick was first made, which usually means real news (injury, weather, etc.) the
# signal has no visibility into. It's a caution flag, not an automatic re-pick.


def load_first_seen() -> dict:
    try:
        with open(FIRST_SEEN_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def update_first_seen(first_seen: dict, game_id: str, spread_line: float, today: str) -> None:
    """Record a game's spread the first time we ever see odds posted for it.
    Never overwritten afterward -- this is the fixed reference point line
    movement gets measured against."""
    if game_id not in first_seen and spread_line is not None:
        first_seen[game_id] = {"spread": spread_line, "date": today}


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
    import datetime
    today = datetime.date.today().isoformat()
    first_seen = load_first_seen()

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

        current_spread = float(g["spread_line"]) if pd.notna(g["spread_line"]) else None
        if has_odds:
            update_first_seen(first_seen, g["game_id"], current_spread, today)

        first_seen_entry = first_seen.get(g["game_id"])
        spread_move = None
        line_moved_significantly = False
        if has_odds and first_seen_entry is not None:
            spread_move = round(current_spread - first_seen_entry["spread"], 2)
            line_moved_significantly = abs(spread_move) >= LINE_MOVE_FLAG_THRESHOLD

        records.append({
            "season": 2026, "week": int(g["week"]), "gameday": g["gameday"], "game_id": g["game_id"],
            "home_team": g["home_team"], "away_team": g["away_team"],
            "spread_line": current_spread,
            "has_odds": bool(has_odds), "bet_side": bet_side, "bet_odds": bet_odds, "no_pick": no_pick,
            "home_signal": round(home_sig, 4), "away_signal": round(away_sig, 4),
            "abs_signal": round(abs(signal_diff), 4),
            "first_seen_spread": first_seen_entry["spread"] if first_seen_entry else None,
            "first_seen_date": first_seen_entry["date"] if first_seen_entry else None,
            "spread_move": spread_move,
            "line_moved_significantly": line_moved_significantly,
        })

    if missing_teams:
        print(f"WARNING: no current predictability state for: {missing_teams} (bye/relocation edge case)")

    # Rank by signal magnitude within each week (presnap-safe -- no odds or outcomes
    # involved) to flag the top-2 highest-conviction picks, same convention as the
    # historical log's is_top2 flag.
    recs_df = pd.DataFrame(records)
    if len(recs_df):
        recs_df["rank_in_week"] = recs_df.groupby("week")["abs_signal"].rank(ascending=False, method="first")
        recs_df["is_top2"] = recs_df["rank_in_week"] <= 2
        records = recs_df.drop(columns=["rank_in_week"]).to_dict("records")

    with open(FIRST_SEEN_PATH, "w") as f:
        json.dump(first_seen, f, indent=2)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} upcoming-game predictions to {OUTPUT_PATH}")

    n_with_odds = sum(r["has_odds"] for r in records)
    print(f"{n_with_odds} of {len(records)} 2026 games have market lines posted so far")

    moved = [r for r in records if r["line_moved_significantly"]]
    if moved:
        print(f"\n{len(moved)} game(s) with a line move >= {LINE_MOVE_FLAG_THRESHOLD} pts since first seen "
              f"(pick direction is NOT re-evaluated -- flagged for manual review only):")
        for r in moved:
            pick_team = r["home_team"] if r["bet_side"] == "home" else r["away_team"]
            print(f"  {r['away_team']} @ {r['home_team']}: pick {pick_team}, "
                  f"{r['first_seen_spread']:+.1f} -> {r['spread_line']:+.1f} ({r['spread_move']:+.1f})")
    print("\nWeek 1 predictions:")
    for r in records:
        if r["week"] == 1:
            pick_team = r["home_team"] if r["bet_side"] == "home" else r["away_team"]
            print(f"  {r['away_team']} @ {r['home_team']}: pick {pick_team} "
                  f"(line {r['spread_line']}, odds {r['bet_odds']})" if r["has_odds"]
                  else f"  {r['away_team']} @ {r['home_team']}: pick {pick_team} (line TBD)")


if __name__ == "__main__":
    main()
