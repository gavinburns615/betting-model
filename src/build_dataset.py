"""Join games/odds/weather with the presnap-safe predictability signal into
one time-indexed, per-game dataset -- the final Stage 1 deliverable.

Team-code reconciliation: nflverse's play-by-play data (which the pass-run
predictability signal is built from) backfills relocated franchises to their
CURRENT code for all historical seasons (Oakland Raiders always "LV", San
Diego Chargers always "LAC"). The schedules/odds dataset instead uses the
code that was actually in use that season ("OAK" through 2019, "SD" through
2016). Both are normalized to the pbp convention here purely as a join key;
the original historical codes are kept for display.

Spread convention (confirmed empirically against moneylines): spread_line is
the home team's IMPLIED margin -- positive means the home team is favored by
that many points, negative means they're the underdog. So:
    home_covers  <=>  (home_score - away_score) > spread_line
regardless of which side is the favorite. A tie between margin and line is a
push (recorded separately, not folded into either side).
"""
import pandas as pd

GAMES_PATH = "data/raw/games_odds.parquet"
SIGNAL_PATH = "data/processed/predictability_signal.parquet"
OUTPUT_PATH = "data/processed/game_dataset.parquet"

TEAM_CODE_FIX = {"OAK": "LV", "SD": "LAC"}

SIGNAL_COLS = [
    "model_confidence_entering", "model_accuracy_entering",
    "pass_rate_extremity_entering", "raw_pass_rate_entering", "n_plays",
]


def normalize_team(col: pd.Series) -> pd.Series:
    return col.replace(TEAM_CODE_FIX)


def main():
    print(f"Loading {GAMES_PATH} and {SIGNAL_PATH}")
    games = pd.read_parquet(GAMES_PATH)
    signal = pd.read_parquet(SIGNAL_PATH)

    games = games.copy()
    games["home_team_join"] = normalize_team(games["home_team"])
    games["away_team_join"] = normalize_team(games["away_team"])

    home_sig = signal[["season", "week", "posteam"] + SIGNAL_COLS].rename(
        columns={"posteam": "home_team_join", **{c: f"home_{c}" for c in SIGNAL_COLS}}
    )
    away_sig = signal[["season", "week", "posteam"] + SIGNAL_COLS].rename(
        columns={"posteam": "away_team_join", **{c: f"away_{c}" for c in SIGNAL_COLS}}
    )

    df = games.merge(home_sig, on=["season", "week", "home_team_join"], how="left")
    df = df.merge(away_sig, on=["season", "week", "away_team_join"], how="left")

    df["home_margin"] = df["home_score"] - df["away_score"]
    df["home_covers"] = df["home_margin"] > df["spread_line"]
    df["away_covers"] = df["home_margin"] < df["spread_line"]
    df["push"] = df["home_margin"] == df["spread_line"]
    df["home_win"] = df["home_margin"] > 0
    df["away_win"] = df["home_margin"] < 0
    df["tie"] = df["home_margin"] == 0

    df["home_minus_away_model_confidence_entering"] = (
        df["home_model_confidence_entering"] - df["away_model_confidence_entering"]
    )

    keep = [
        "game_id", "season", "week", "gameday", "home_team", "away_team",
        "home_score", "away_score", "home_margin",
        "spread_line", "home_spread_odds", "away_spread_odds",
        "total_line", "over_odds", "under_odds",
        "home_moneyline", "away_moneyline",
        "roof", "surface", "temp", "wind",
        "home_covers", "away_covers", "push", "home_win", "away_win", "tie",
        "home_model_confidence_entering", "away_model_confidence_entering",
        "home_minus_away_model_confidence_entering",
        "home_model_accuracy_entering", "away_model_accuracy_entering",
        "home_pass_rate_extremity_entering", "away_pass_rate_extremity_entering",
        "home_raw_pass_rate_entering", "away_raw_pass_rate_entering",
        "home_n_plays", "away_n_plays",
    ]
    df = df[keep].sort_values(["season", "week", "game_id"]).reset_index(drop=True)

    n_no_signal = df["home_model_confidence_entering"].isna().sum()
    print(f"{len(df):,} games total. {n_no_signal:,} have no predictability signal yet "
          f"(2016-2017, before the signal's 2018 start, or week-1 openers) -- "
          f"these will need to be dropped or handled explicitly at the strategy stage.")

    df.to_parquet(OUTPUT_PATH)
    print(f"Saved {len(df):,} games to {OUTPUT_PATH}")

    print("\n--- Sample rows (mid-season 2023, full signal available) ---")
    sample = df[(df["season"] == 2023) & (df["week"] == 8)]
    cols = ["game_id", "home_team", "away_team", "spread_line", "home_moneyline", "away_moneyline",
            "home_score", "away_score", "home_covers",
            "home_model_confidence_entering", "away_model_confidence_entering"]
    print(sample[cols].to_string(index=False))


if __name__ == "__main__":
    main()
