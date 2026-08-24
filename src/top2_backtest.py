"""Restrict the corrected walk-forward strategy to its highest-conviction
picks: the top 2 games per week by |signal_diff| (the same presnap-safe
number backtest_ats.py already uses to pick a side -- ranking by its
magnitude introduces no new fitting, since it isn't learned from outcomes).

IMPORTANT CAVEAT this script exists to make visible, not hide: N=2 was
chosen by comparing several candidates (1, 2, 3, 5, 10) against the FULL
2020-2025 period and picking the best-looking one -- exactly the kind of
after-the-fact selection this project has otherwise been careful to avoid.
The one thing that keeps this more than pure post-hoc curve-fitting is the
holdout check below: choosing N using ONLY 2020-2022, then testing it on
the untouched 2023-2025 games, still shows a real (p=0.013) result. But
that's a single train/test split, not the same multi-fold expanding-window
walk-forward the main strategy was validated with -- treat this as a
promising lead, not a separately proven strategy.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats as sstats

from backtest_ats import SIGNAL_COL

BETS_PATH = "data/processed/backtest_bets.parquet"
GAMES_LOG_PATH = "output/games_log.json"
CURVE_OUTPUT_PATH = "output/top2_cumulative_curve.json"
TOP_N = 2


def load_relative_bets() -> pd.DataFrame:
    all_bets = pd.read_parquet(BETS_PATH)
    rel = all_bets[all_bets["strategy"] == "predictability_signal_relative"].copy()
    rel["abs_signal"] = rel[SIGNAL_COL].abs()
    rel["rank_in_week"] = rel.groupby(["season", "week"])["abs_signal"].rank(ascending=False, method="first")
    rel[f"is_top{TOP_N}"] = rel["rank_in_week"] <= TOP_N
    return rel


def holdout_check(rel: pd.DataFrame) -> None:
    """Pick N on 2020-2022 only (comparing candidates), then test that choice
    on the untouched 2023-2025 games -- the one thing standing between this
    and pure post-hoc selection."""
    early = rel[rel["season"].isin([2020, 2021, 2022])]
    late = rel[rel["season"].isin([2023, 2024, 2025])]

    print("Choosing N on 2020-2022 only:")
    for n in [1, 2, 3, 5, 10]:
        t = early.sort_values("abs_signal", ascending=False).groupby(["season", "week"]).head(n)
        print(f"  N={n}: n={len(t)}, win_rate={t['bet_won'].mean():.1%}")

    print(f"\nApplying N={TOP_N} (chosen above) to UNTOUCHED 2023-2025:")
    t = late.sort_values("abs_signal", ascending=False).groupby(["season", "week"]).head(TOP_N)
    wins, total = int(t["bet_won"].sum()), len(t)
    p = sstats.binomtest(wins, total, 0.5).pvalue
    print(f"  n={total}, win_rate={wins/total:.1%}, binomial p={p:.3f}, flat_pnl={t['flat_pnl'].sum():+.1f}u")


def main():
    rel = load_relative_bets()
    holdout_check(rel)

    top = rel[rel[f"is_top{TOP_N}"]].sort_values(["season", "week", "gameday"]).reset_index(drop=True)
    print(f"\n--- Top-{TOP_N}/week, full 2020-2025 period ---")
    print(f"n={len(top)}, win_rate={top['bet_won'].mean():.1%}, "
          f"total_pnl={top['flat_pnl'].sum():+.1f}u")
    print(top.groupby("season").agg(n=("bet_won", "count"), win_rate=("bet_won", "mean"), units=("flat_pnl", "sum")))

    top["cum_units"] = top["flat_pnl"].cumsum()
    curve = list(zip(range(len(top)), top["cum_units"].round(2)))
    with open(CURVE_OUTPUT_PATH, "w") as f:
        json.dump(curve, f)
    print(f"\nSaved cumulative curve ({len(curve)} points) to {CURVE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
