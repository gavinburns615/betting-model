"""Full metrics suite: ROI, Sharpe, max drawdown, weekly/monthly breakdown,
computed identically for the strategy and both baselines so they're directly
comparable."""
import numpy as np
import pandas as pd


def summary_stats(bets: pd.DataFrame) -> dict:
    n = len(bets)
    wins = int(bets["bet_won"].sum())
    win_rate = wins / n
    flat_total_pnl = bets["flat_pnl"].sum()
    flat_roi = bets["flat_pnl"].mean()

    se = np.sqrt(0.5 * 0.5 / n)
    z = (win_rate - 0.5) / se

    out = {
        "n_bets": n,
        "wins": wins,
        "losses": n - wins,
        "win_rate": win_rate,
        "win_rate_vs_50pct_z": float(z),
        "flat_total_pnl_units": float(flat_total_pnl),
        "flat_roi_per_bet": float(flat_roi),
    }

    if "kelly_bankroll" in bets.columns and bets["kelly_bankroll"].notna().all():
        start = 1000.0
        end = bets["kelly_bankroll"].iloc[-1]
        out["kelly_start_bankroll"] = start
        out["kelly_end_bankroll"] = float(end)
        out["kelly_total_return_pct"] = float((end - start) / start)

    return out


def cumulative_curve(bets: pd.DataFrame) -> pd.DataFrame:
    df = bets.sort_values(["season", "week", "game_id"]).reset_index(drop=True).copy()
    df["cum_flat_pnl"] = df["flat_pnl"].cumsum()
    return df


def sharpe_and_drawdown(bets: pd.DataFrame) -> dict:
    df = cumulative_curve(bets)

    weekly = df.groupby(["season", "week"])["flat_pnl"].mean()
    weekly_sharpe = weekly.mean() / weekly.std(ddof=1) if weekly.std(ddof=1) > 0 else float("nan")
    weekly_sharpe_annualized = weekly_sharpe * np.sqrt(18)  # ~18 weeks/NFL season

    running_peak = df["cum_flat_pnl"].cummax()
    drawdown_units = df["cum_flat_pnl"] - running_peak
    max_dd_units = drawdown_units.min()

    out = {
        "weekly_sharpe_raw": float(weekly_sharpe),
        "weekly_sharpe_season_annualized": float(weekly_sharpe_annualized),
        "max_drawdown_units_flat": float(max_dd_units),
        "n_weeks": int(len(weekly)),
    }

    if "kelly_bankroll" in df.columns and df["kelly_bankroll"].notna().all():
        peak = df["kelly_bankroll"].cummax()
        dd_pct = (df["kelly_bankroll"] - peak) / peak
        out["max_drawdown_pct_kelly"] = float(dd_pct.min())

    return out


def weekly_breakdown(bets: pd.DataFrame) -> pd.DataFrame:
    g = bets.groupby(["season", "week"]).agg(
        n_bets=("bet_won", "count"),
        win_rate=("bet_won", "mean"),
        flat_pnl=("flat_pnl", "sum"),
    ).reset_index()
    return g


def monthly_breakdown(bets: pd.DataFrame) -> pd.DataFrame:
    df = bets.copy()
    df["month"] = pd.to_datetime(df["gameday"]).dt.to_period("M").astype(str)
    g = df.groupby("month").agg(
        n_bets=("bet_won", "count"),
        win_rate=("bet_won", "mean"),
        flat_pnl=("flat_pnl", "sum"),
    ).reset_index()
    return g


def season_breakdown(bets: pd.DataFrame) -> pd.DataFrame:
    g = bets.groupby("season").agg(
        n_bets=("bet_won", "count"),
        win_rate=("bet_won", "mean"),
        flat_pnl=("flat_pnl", "sum"),
    ).reset_index()
    return g
