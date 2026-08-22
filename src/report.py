"""Stage 4: full metrics report + cumulative bankroll chart + baseline sanity
check, run over the walk-forward backtest results from backtest_ats.py."""
import json

import matplotlib.pyplot as plt
import pandas as pd

from metrics import (
    cumulative_curve, monthly_breakdown, season_breakdown,
    sharpe_and_drawdown, summary_stats, weekly_breakdown,
)

BETS_PATH = "data/processed/backtest_bets.parquet"
CHART_PATH = "output/cumulative_bankroll.png"
REPORT_JSON_PATH = "output/report.json"

STRATEGIES = ["predictability_signal", "predictability_signal_relative", "baseline_home", "baseline_coinflip"]
LABELS = {
    "predictability_signal": "Predictability signal (original, broken)",
    "predictability_signal_relative": "Predictability signal (corrected)",
    "baseline_home": "Baseline: always bet home",
    "baseline_coinflip": "Baseline: coin flip",
}


def build_report() -> dict:
    all_bets = pd.read_parquet(BETS_PATH)
    report = {}
    for strat in STRATEGIES:
        b = all_bets[all_bets["strategy"] == strat]
        report[strat] = {
            **summary_stats(b),
            **sharpe_and_drawdown(b),
        }
    return report, all_bets


def print_report(report: dict) -> None:
    print("=" * 78)
    print("FULL METRICS REPORT (walk-forward, out-of-sample, 2020-2025 test seasons)")
    print("=" * 78)
    for strat in STRATEGIES:
        r = report[strat]
        print(f"\n--- {LABELS[strat]} ---")
        print(f"  Bets:            {r['n_bets']:,}  ({r['wins']}-{r['losses']})")
        print(f"  Win rate:        {r['win_rate']:.1%}  (z vs 50% = {r['win_rate_vs_50pct_z']:+.2f})")
        print(f"  Flat ROI/bet:    {r['flat_roi_per_bet']:+.2%}")
        print(f"  Flat total P/L:  {r['flat_total_pnl_units']:+.2f} units")
        if "kelly_end_bankroll" in r:
            print(f"  Kelly bankroll:  ${r['kelly_start_bankroll']:.0f} -> ${r['kelly_end_bankroll']:.2f} "
                  f"({r['kelly_total_return_pct']:+.1%})")
            print(f"  Kelly max drawdown: {r['max_drawdown_pct_kelly']:.1%}")
        print(f"  Weekly Sharpe (raw): {r['weekly_sharpe_raw']:+.3f}  "
              f"(season-annualized: {r['weekly_sharpe_season_annualized']:+.3f})")
        print(f"  Max drawdown (flat, units): {r['max_drawdown_units_flat']:.2f}")

    print("\n" + "=" * 78)
    print("SANITY CHECK: do the baselines show a spurious edge?")
    print("=" * 78)
    home_wr = report["baseline_home"]["win_rate"]
    coin_wr = report["baseline_coinflip"]["win_rate"]
    home_z = report["baseline_home"]["win_rate_vs_50pct_z"]
    coin_z = report["baseline_coinflip"]["win_rate_vs_50pct_z"]
    leak = abs(home_z) > 2 or abs(coin_z) > 2
    print(f"  baseline_home win rate:     {home_wr:.1%} (z={home_z:+.2f})")
    print(f"  baseline_coinflip win rate: {coin_wr:.1%} (z={coin_z:+.2f})")
    if leak:
        print("  *** WARNING: a baseline shows a statistically significant edge (|z|>2). ***")
        print("  *** This points to a LEAK in the harness -- do not trust the strategy result. ***")
    else:
        print("  OK: neither baseline shows a significant edge (|z|<2). No leak detected.")

    strat_wr = report["predictability_signal"]["win_rate"]
    strat_z = report["predictability_signal"]["win_rate_vs_50pct_z"]
    print(f"\n  predictability_signal win rate: {strat_wr:.1%} (z={strat_z:+.2f})")
    if abs(strat_z) < 2:
        print("  -> Not statistically distinguishable from a coin flip out-of-sample.")
        print("  -> The in-sample tune correlation (see fold_summary.json) did not generalize.")
        print("  -> CONCLUSION: no exploitable edge found. Do not deploy this strategy as-is.")
    else:
        print("  -> Statistically distinguishable from 50% out-of-sample (|z|>2) -- investigate further")
        print("     (check it survives out past the vig, and isn't itself a leak).")


def make_chart(all_bets: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    for strat in STRATEGIES:
        b = all_bets[all_bets["strategy"] == strat]
        c = cumulative_curve(b)
        x = range(len(c))
        ax.plot(x, c["cum_flat_pnl"], label=LABELS[strat], linewidth=1.6)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("Cumulative flat-stake P/L (1 unit/bet)")
    ax.set_xlabel("Bet # (chronological, 2020-2025)")
    ax.set_ylabel("Cumulative units")
    ax.legend()

    ax = axes[1]
    strat_bets = all_bets[all_bets["strategy"] == "predictability_signal"]
    c = cumulative_curve(strat_bets)
    x = range(len(c))
    ax.plot(x, c["kelly_bankroll"], color="tab:blue", linewidth=1.6)
    ax.axhline(1000, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("Predictability signal: Kelly bankroll ($1,000 start, 25% cap)")
    ax.set_xlabel("Bet # (chronological, 2020-2025)")
    ax.set_ylabel("Bankroll ($)")

    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=150)
    print(f"\nSaved chart to {CHART_PATH}")


def main():
    report, all_bets = build_report()
    print_report(report)
    make_chart(all_bets)

    print("\n--- Season breakdown (predictability_signal) ---")
    strat_bets = all_bets[all_bets["strategy"] == "predictability_signal"]
    print(season_breakdown(strat_bets).to_string(index=False))

    print("\n--- Monthly breakdown (predictability_signal) ---")
    print(monthly_breakdown(strat_bets).to_string(index=False))

    with open(REPORT_JSON_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved full report to {REPORT_JSON_PATH}")


if __name__ == "__main__":
    main()
