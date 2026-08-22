"""American-odds payout math shared by the strategy and baseline harnesses."""
import numpy as np


def american_to_decimal(odds: float) -> float:
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)


def american_to_implied_prob(odds: float) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def kelly_fraction(win_prob: float, odds: float, cap: float = 0.25) -> float:
    """Fraction of bankroll to stake. b = net decimal odds (payout per $1 staked,
    excluding the stake itself). Negative/zero-edge bets get 0. Capped (default 25%
    of bankroll) since raw Kelly on a noisy win-prob estimate is dangerously
    aggressive -- a standard 'fractional Kelly' safeguard, not part of the edge
    hypothesis itself."""
    b = american_to_decimal(odds) - 1
    f = (win_prob * (b + 1) - 1) / b
    return float(np.clip(f, 0, cap))
