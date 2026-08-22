"""Walk-forward ATS (against-the-spread) backtest of the predictability-signal
strategy, with a baseline sanity check run through the identical harness.

STRATEGY (locked at each fold from TUNE data only, never from the test season
itself -- this is the walk-forward guarantee):
  For each game, signal_diff = home_model_confidence_entering -
  away_model_confidence_entering (both presnap-safe, see
  build_predictability_signal.py). Fit a 1-variable logistic regression of
  home_covers ~ signal_diff on every prior season's games. Apply it to the
  upcoming season's games to get a predicted probability the home team
  covers; bet whichever side that implies, at that game's actual market odds
  for that side. The regression's sign and magnitude -- i.e. which
  direction the edge even runs -- is discovered by the model at each fold,
  not assumed up front.

FOLDS: test season S uses every earlier season back to 2018 as its tune set
(expanding window, matching pass-run's own backtest.py convention). Only
seasons with >=2 tune seasons behind them are tested, so folds start at 2020.

BASELINES (run through the exact same fold structure, same bet-or-not gate
[only games with full signal available], same staking math): always bet the
home team ATS, and seeded coin-flip picks. Neither should show a durable
edge -- if one does, that's a leakage red flag, not a strategy worth using.
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy import stats as sstats

from odds_utils import american_to_decimal, kelly_fraction

DATASET_PATH = "data/processed/game_dataset.parquet"
OUTPUT_PATH = "data/processed/backtest_bets.parquet"
FOLD_SUMMARY_PATH = "data/processed/fold_summary.json"

MIN_TUNE_SEASONS = 2
SIGNAL_COL = "home_minus_away_model_confidence_entering"
RANDOM_STATE = 0


def load_bettable_games() -> pd.DataFrame:
    df = pd.read_parquet(DATASET_PATH)
    df = df[df["season"] >= 2018].copy()
    df = df.dropna(subset=[SIGNAL_COL, "spread_line", "home_spread_odds", "away_spread_odds"])
    df = df[~df["push"]]  # pushes are excluded from ATS win-rate/ROI by convention
    df["gameday"] = pd.to_datetime(df["gameday"])
    return df.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def fit_direction(tune_df: pd.DataFrame) -> tuple[LogisticRegression, dict]:
    X = tune_df[[SIGNAL_COL]].values
    y = tune_df["home_covers"].astype(int).values
    model = LogisticRegression()
    model.fit(X, y)

    r, p = sstats.pearsonr(tune_df[SIGNAL_COL], tune_df["home_covers"].astype(int))
    stats = {
        "n_tune_games": int(len(tune_df)),
        "coef": float(model.coef_[0][0]),
        "intercept": float(model.intercept_[0]),
        "pearson_r": float(r),
        "pearson_p": float(p),
    }
    return model, stats


def make_bets(test_df: pd.DataFrame, model: LogisticRegression) -> pd.DataFrame:
    """NOTE: kept for reference / comparison only -- see make_bets_relative below.
    This absolute-probability version turned out to be broken: the fitted
    intercept (which just encodes each fold's tune-window home-cover base
    rate -- itself small-sample noise) dominates the coefficient on the
    actual signal, so predicted P(home covers) never crosses 0.5 in any
    fold. The result degenerates into a disguised 'always bet away' rule
    that never actually varies with the signal. Left in so the difference
    is visible in the report."""
    df = test_df.copy()
    home_cover_prob = model.predict_proba(df[[SIGNAL_COL]].values)[:, 1]
    df["model_home_cover_prob"] = home_cover_prob

    bet_home = home_cover_prob >= 0.5
    df["bet_side"] = np.where(bet_home, "home", "away")
    df["bet_prob"] = np.where(bet_home, home_cover_prob, 1 - home_cover_prob)
    df["bet_odds"] = np.where(bet_home, df["home_spread_odds"], df["away_spread_odds"])
    df["bet_won"] = np.where(bet_home, df["home_covers"], df["away_covers"])
    return df


def make_bets_relative(test_df: pd.DataFrame, coef_sign: float) -> pd.DataFrame:
    """The corrected strategy: bet on the SIGN of signal_diff alone, ignoring
    the fold's fitted intercept entirely. This tests the actual hypothesis --
    does relative predictability between the two teams predict who covers --
    without letting a noisy tune-window base rate silently override it.
    coef_sign is the sign of the tune-fold's fitted coefficient (still
    walk-forward: discovered from prior seasons only, never the test season).
    bet_prob for Kelly sizing comes from a simple monotonic mapping of
    |signal_diff|'s rank within the tune set to a win-probability estimate --
    see fit_relative_prob_map. Games with signal_diff exactly 0 are dropped
    (no directional information)."""
    df = test_df[test_df[SIGNAL_COL] != 0].copy()
    home_favored = (df[SIGNAL_COL] * coef_sign) > 0
    df["bet_side"] = np.where(home_favored, "home", "away")
    df["bet_odds"] = np.where(home_favored, df["home_spread_odds"], df["away_spread_odds"])
    df["bet_won"] = np.where(home_favored, df["home_covers"], df["away_covers"])
    return df


def tune_win_rate_for_rule(tune_df: pd.DataFrame, coef_sign: float) -> float:
    """Walk-forward win-probability estimate for the relative-sign rule, used
    only for Kelly sizing: what fraction of TUNE-set games would this exact
    rule have won, had it been applied there. A single constant per fold
    (not a magnitude-based curve) -- deliberately simple, to avoid smuggling
    a second round of overfitting into the Kelly stake itself."""
    sub = tune_df[tune_df[SIGNAL_COL] != 0]
    home_favored = (sub[SIGNAL_COL] * coef_sign) > 0
    won = np.where(home_favored, sub["home_covers"], sub["away_covers"])
    return float(np.mean(won))


def make_baseline_home_bets(test_df: pd.DataFrame) -> pd.DataFrame:
    df = test_df.copy()
    df["bet_side"] = "home"
    df["bet_prob"] = np.nan  # baseline: no model probability, flat stakes only
    df["bet_odds"] = df["home_spread_odds"]
    df["bet_won"] = df["home_covers"]
    return df


def make_baseline_coinflip_bets(test_df: pd.DataFrame, rng: np.random.RandomState) -> pd.DataFrame:
    df = test_df.copy()
    pick_home = rng.rand(len(df)) < 0.5
    df["bet_side"] = np.where(pick_home, "home", "away")
    df["bet_prob"] = np.nan
    df["bet_odds"] = np.where(pick_home, df["home_spread_odds"], df["away_spread_odds"])
    df["bet_won"] = np.where(pick_home, df["home_covers"], df["away_covers"])
    return df


def run_walk_forward(strategy: str) -> tuple[pd.DataFrame, list[dict]]:
    df = load_bettable_games()
    seasons = sorted(df["season"].unique())
    test_seasons = [s for i, s in enumerate(seasons) if i >= MIN_TUNE_SEASONS]

    rng = np.random.RandomState(RANDOM_STATE)
    all_bets, fold_stats = [], []
    for season in test_seasons:
        tune_df = df[df["season"] < season]
        test_df = df[df["season"] == season]

        if strategy == "predictability_signal":
            model, stats_row = fit_direction(tune_df)
            stats_row["season"] = int(season)
            fold_stats.append(stats_row)
            bets = make_bets(test_df, model)
        elif strategy == "predictability_signal_relative":
            model, stats_row = fit_direction(tune_df)
            coef_sign = 1.0 if stats_row["coef"] >= 0 else -1.0
            stats_row["season"] = int(season)
            stats_row["coef_sign"] = coef_sign
            stats_row["tune_rule_win_rate"] = tune_win_rate_for_rule(tune_df, coef_sign)
            fold_stats.append(stats_row)
            bets = make_bets_relative(test_df, coef_sign)
            bets["bet_prob"] = stats_row["tune_rule_win_rate"]
        elif strategy == "baseline_home":
            bets = make_baseline_home_bets(test_df)
        elif strategy == "baseline_coinflip":
            bets = make_baseline_coinflip_bets(test_df, rng)
        else:
            raise ValueError(strategy)

        all_bets.append(bets)

    result = pd.concat(all_bets, ignore_index=True)
    result["strategy"] = strategy
    return result, fold_stats


def add_pnl_columns(bets: pd.DataFrame, kelly_cap: float = 0.25) -> pd.DataFrame:
    bets = bets.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    dec_odds = bets["bet_odds"].apply(american_to_decimal)
    bets["decimal_odds"] = dec_odds
    bets["flat_pnl"] = np.where(bets["bet_won"], dec_odds - 1, -1.0)  # 1-unit flat stake

    if bets["bet_prob"].notna().all():
        bets["kelly_frac"] = [
            kelly_fraction(p, o, cap=kelly_cap) for p, o in zip(bets["bet_prob"], bets["bet_odds"])
        ]
    else:
        bets["kelly_frac"] = np.nan
    return bets


def simulate_kelly_bankroll(bets: pd.DataFrame, start_bankroll: float = 1000.0) -> pd.DataFrame:
    bankroll = start_bankroll
    curve = []
    for _, row in bets.iterrows():
        stake = bankroll * row["kelly_frac"] if pd.notna(row["kelly_frac"]) else 0.0
        pnl = stake * (row["decimal_odds"] - 1) if row["bet_won"] else -stake
        bankroll += pnl
        curve.append(bankroll)
    bets = bets.copy()
    bets["kelly_bankroll"] = curve
    return bets


def main():
    print("=" * 70)
    print("STRATEGY: predictability signal (walk-forward, direction locked per fold)")
    print("=" * 70)
    strat_bets, fold_stats = run_walk_forward("predictability_signal")
    strat_bets = add_pnl_columns(strat_bets)
    strat_bets = simulate_kelly_bankroll(strat_bets)

    print("\n--- Per-fold tune-set fit (n games, coef sign, correlation, p-value) ---")
    for s in fold_stats:
        print(f"  test={s['season']}  tune_n={s['n_tune_games']:4d}  "
              f"coef={s['coef']:+.3f}  pearson_r={s['pearson_r']:+.4f}  p={s['pearson_p']:.4f}")

    print("\n" + "=" * 70)
    print("STRATEGY (CORRECTED): relative sign of signal_diff, intercept removed")
    print("=" * 70)
    rel_bets, rel_fold_stats = run_walk_forward("predictability_signal_relative")
    rel_bets = add_pnl_columns(rel_bets)
    rel_bets = simulate_kelly_bankroll(rel_bets)
    print("\n--- Per-fold (coef sign, tune-set win rate of the rule) ---")
    for s in rel_fold_stats:
        print(f"  test={s['season']}  tune_n={s['n_tune_games']:4d}  "
              f"coef_sign={s['coef_sign']:+.0f}  tune_rule_win_rate={s['tune_rule_win_rate']:.1%}")

    print("\n--- BASELINE: always bet home team ATS ---")
    home_bets, _ = run_walk_forward("baseline_home")
    home_bets = add_pnl_columns(home_bets)

    print("--- BASELINE: coin flip ---")
    coin_bets, _ = run_walk_forward("baseline_coinflip")
    coin_bets = add_pnl_columns(coin_bets)

    all_results = pd.concat([strat_bets, rel_bets, home_bets, coin_bets], ignore_index=True)
    all_results.to_parquet(OUTPUT_PATH)
    with open(FOLD_SUMMARY_PATH, "w") as f:
        json.dump(fold_stats, f, indent=2)
    with open(FOLD_SUMMARY_PATH.replace(".json", "_relative.json"), "w") as f:
        json.dump(rel_fold_stats, f, indent=2)

    print(f"\nSaved {len(all_results):,} bet rows to {OUTPUT_PATH}")
    print(f"Saved fold-by-fold tune stats to {FOLD_SUMMARY_PATH}")

    for name, b in [("predictability_signal", strat_bets), ("predictability_signal_relative", rel_bets),
                     ("baseline_home", home_bets), ("baseline_coinflip", coin_bets)]:
        wr = b["bet_won"].mean()
        roi = b["flat_pnl"].mean()
        print(f"\n{name}: n={len(b):,}  win_rate={wr:.1%}  flat_ROI/bet={roi:+.1%}")


if __name__ == "__main__":
    main()
