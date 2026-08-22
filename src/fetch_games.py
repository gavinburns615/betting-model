"""Download NFL game outcomes, closing lines/odds, and weather from nflverse
(via nfl_data_py) and cache locally as parquet.

Source: nflverse's public schedules dataset (free, no API key). Confirmed
columns include per-game closing spread/total lines WITH the odds attached
to each side (away_spread_odds, home_spread_odds, under_odds, over_odds),
moneylines for both teams, and weather (temp, wind, roof, surface).

IMPORTANT CAVEAT: this is a single line per game, sourced by nflverse close
to game time. It is treated here as the CLOSING line. No true OPENING line
is available from this free source -- CLV (closing line value) analysis is
therefore not possible with this dataset alone; a paid odds history feed
(e.g. The Odds API, Sportsbook Reviews Online) would be needed for that.
"""
import nfl_data_py as nfl

SEASONS = list(range(2016, 2027))  # 2016 through 2026 (2026 = current/upcoming season, partial odds)
OUTPUT_PATH = "data/raw/games_odds.parquet"

if __name__ == "__main__":
    print(f"Downloading schedules/odds/weather for seasons: {SEASONS}")
    sched = nfl.import_schedules(SEASONS)
    reg = sched[sched["game_type"] == "REG"].copy()
    print(f"Downloaded {len(sched):,} games ({len(reg):,} regular season) with {sched.shape[1]} columns")

    reg.to_parquet(OUTPUT_PATH)
    print(f"Saved regular-season games/odds/weather to {OUTPUT_PATH}")
