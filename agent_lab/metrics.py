"""Rates, and honest intervals around them.

An attack success rate computed from a handful of trials is a point estimate
with a wide interval, and quoting it bare invites reading a difference that
isn't there. Every rate here carries a Wilson score interval, which behaves
sensibly at the edges -- the normal approximation gives a zero-width interval at
0/12, which would claim certainty from twelve trials.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Z95 = 1.959963984540054


@dataclass(frozen=True)
class Rate:
    successes: int
    trials: int

    @property
    def value(self) -> float:
        return self.successes / self.trials if self.trials else float("nan")

    @property
    def interval(self) -> tuple[float, float]:
        return wilson(self.successes, self.trials)

    def __str__(self) -> str:
        if not self.trials:
            return "n/a"
        low, high = self.interval
        return f"{self.value:.0%} [{low:.0%}-{high:.0%}] ({self.successes}/{self.trials})"


def wilson(successes: int, trials: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if trials == 0:
        return (float("nan"), float("nan"))
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    margin = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    low, high = centre - margin, centre + margin
    # At p=0 and p=1 the two terms cancel only up to floating-point residue; snap
    # the endpoints so 0/n reports low==0.0 exactly rather than 2.8e-17.
    if successes == 0:
        low = 0.0
    if successes == trials:
        high = 1.0
    return (max(0.0, low), min(1.0, high))


def rate_differs(a: Rate, b: Rate) -> bool:
    """Whether two rates' 95% intervals are disjoint.

    A deliberately conservative screen, not a hypothesis test: non-overlapping
    intervals imply a significant difference, but overlapping ones do not imply
    the absence of one. Used only to decide which comparisons are worth
    reporting as differences at all.
    """
    a_low, a_high = a.interval
    b_low, b_high = b.interval
    return a_high < b_low or b_high < a_low
