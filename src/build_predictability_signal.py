"""Turn the pass-run classifier's play-level backtest predictions into a
per-team, per-game PREDICTABILITY SIGNAL that is safe to use as a betting
input -- i.e. known before that game's kickoff.

Source: ~/pass-run/data/backtest_predictions.parquet -- one row per play,
2018-2025, with pred_prob = the classifier's presnap pass probability for
that specific play. Critically, this file was built by backtest.py's
SEASON-LEVEL walk-forward: every play's pred_prob comes from a model that
was trained only on seasons strictly before that play's season. So no
season's pred_prob values benefit from that season's own data.

That per-season guarantee is NOT enough on its own for a betting signal,
though: using week 5's actual plays to bet on week 5's game would still be
using information (how the game was actually played) that doesn't exist
before kickoff. So this script adds a second, game-level walk-forward on
top: for each team-game, first summarize that game's plays into a
"predictability" score, then compute a recency-weighted (EWMA, span=8,
matching build_features.py's RECENCY_SPAN) average of that score across the
team's PRIOR games only, shifted by one row within each team-season --
identical machinery to the "_enter" pattern the pass-run repo already uses
for its own presnap-safe features. Resets at each season boundary (week 1
has no prior data -> NaN, documented not imputed here; caller decides fill).

Two signals are produced per team-game, both entering-only:
  - model_confidence_entering: mean |pred_prob - 0.5| over the team's plays
    in prior games -- how confidently the SITUATIONAL model (which already
    knows down/distance/personnel/tendency) could call this team's plays.
    High = offense is predictable even accounting for game situation.
  - pass_rate_extremity_entering: |raw pass rate - league average pass rate|
    over prior games -- a blunter "run-heavy or pass-heavy overall" measure,
    kept as a simpler comparison point against the situational signal above.
"""
import pandas as pd

PREDICTIONS_PATH = "/Users/gavinburns/pass-run/data/backtest_predictions.parquet"
OUTPUT_PATH = "data/processed/predictability_signal.parquet"

RECENCY_SPAN = 8  # games; matches pass-run's build_features.py RECENCY_SPAN
LEAGUE_AVG_PASS_RATE = 0.58  # matches pass-run's build_features.py constant


def per_game_scores(preds: pd.DataFrame) -> pd.DataFrame:
    g = preds.groupby(["season", "week", "game_id", "posteam"])
    out = g.apply(
        lambda d: pd.Series({
            "n_plays": len(d),
            "model_confidence": (d["pred_prob"] - 0.5).abs().mean(),
            "model_accuracy": d["correct"].mean(),
            "raw_pass_rate": d["is_pass"].mean(),
        }),
        include_groups=False,
    ).reset_index()
    out["pass_rate_extremity"] = (out["raw_pass_rate"] - LEAGUE_AVG_PASS_RATE).abs()
    return out


def add_entering_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["season", "posteam", "week"]).copy()
    g = df.groupby(["season", "posteam"], sort=False)
    for col in ["model_confidence", "model_accuracy", "pass_rate_extremity", "raw_pass_rate"]:
        through = g[col].transform(lambda s: s.ewm(span=RECENCY_SPAN, adjust=False).mean())
        df[f"{col}_entering"] = through.groupby([df["season"], df["posteam"]]).shift(1)
    return df


def main():
    print(f"Loading {PREDICTIONS_PATH}")
    preds = pd.read_parquet(PREDICTIONS_PATH)
    print(f"Loaded {len(preds):,} plays, seasons {sorted(preds['season'].unique())}")

    print("Summarizing to per-team-per-game predictability scores...")
    scores = per_game_scores(preds)

    print("Computing presnap-safe (entering-only) recency-weighted values...")
    scores = add_entering_values(scores)

    n_missing = scores["model_confidence_entering"].isna().sum()
    print(f"{n_missing:,} of {len(scores):,} team-games have no prior-game history yet "
          f"(season openers) -- entering values are NaN there, by design.")

    scores.to_parquet(OUTPUT_PATH)
    print(f"Saved {len(scores):,} team-game rows to {OUTPUT_PATH}")

    print("\n--- Sample (first team-games with full history, mid-season) ---")
    sample = scores.dropna(subset=["model_confidence_entering"]).sort_values(["season", "week"]).head(8)
    cols = ["season", "week", "game_id", "posteam", "n_plays",
            "model_confidence_entering", "model_accuracy_entering",
            "pass_rate_extremity_entering", "raw_pass_rate_entering"]
    print(sample[cols].to_string(index=False))


if __name__ == "__main__":
    main()
