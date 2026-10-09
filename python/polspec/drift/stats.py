"""The tests behind drift's statistical findings, without SciPy.

Drift reports a rate, a set of frequencies or a distribution as moved only
when the move is both *significant* -- unlikely to be sampling noise at the
number of rows seen -- and *large*: bigger than a minimum effect. Each
function here answers the first half with a p-value: the probability of a
difference at least this large if the declaration were exactly right.

- `binomial_p`: a rate (nulls, NaN), by the exact two-sided binomial test.
- `chi_square_p`: category frequencies, by Pearson's goodness of fit.
- `ks_p`: two samples, by the asymptotic two-sided Kolmogorov-Smirnov test.
- `unseen_p`: the chance a value with a given share never appears.

Each matches SciPy's function of the same purpose (`binomtest`,
`chisquare`, `ks_2samp(method="asymp")`); the test suite holds them to it.

Internal: not part of the public API.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = ["binomial_p", "chi_square_p", "ks_p", "total_variation", "unseen_p"]

# SciPy's tolerance for "as likely as the observed count" in the two-sided
# binomial test: counts whose probability is within this relative margin of
# the observed one's count as equally extreme.
_RELATIVE_TIE = 1 + 1e-7


def binomial_p(successes: int, trials: int, p: float) -> float:
    """The two-sided p-value of `successes` in `trials` at rate `p`.

    Exact: the probability of every count no more likely than the observed
    one. Each tail is summed from its edge outward and stops once its terms
    no longer register, so a million trials costs a few thousand terms.
    """
    if trials <= 0:
        return 1.0
    if p <= 0.0:
        return 1.0 if successes == 0 else 0.0
    if p >= 1.0:
        return 1.0 if successes == trials else 0.0

    log_p, log_q = math.log(p), math.log1p(-p)
    log_n = math.lgamma(trials + 1)

    def log_pmf(k: int) -> float:
        return (
            log_n
            - math.lgamma(k + 1)
            - math.lgamma(trials - k + 1)
            + k * log_p
            + (trials - k) * log_q
        )

    mode = min(trials, max(0, math.floor((trials + 1) * p)))
    observed = log_pmf(successes)
    limit = observed + math.log(_RELATIVE_TIE)
    if log_pmf(mode) <= limit:
        return 1.0  # the observed count is as likely as the likeliest

    if successes < mode:
        near = _tail_sum(log_pmf, successes, -1, trials)
        # The first count above the mode no likelier than the observed one.
        start = _first_at_most(log_pmf, mode, trials, limit)
        far = _tail_sum(log_pmf, start, +1, trials) if start is not None else 0.0
    else:
        near = _tail_sum(log_pmf, successes, +1, trials)
        start = _last_at_most(log_pmf, 0, mode, limit)
        far = _tail_sum(log_pmf, start, -1, trials) if start is not None else 0.0
    return min(1.0, near + far)


def _tail_sum(log_pmf, start: int, step: int, trials: int) -> float:
    """The probability of `start` and every count beyond it, away from the
    mode, stopping once a term no longer changes the sum."""
    total, k = 0.0, start
    while 0 <= k <= trials:
        term = math.exp(log_pmf(k))
        total += term
        if term < total * 1e-17:
            break
        k += step
    return total


def _first_at_most(log_pmf, lo: int, hi: int, limit: float) -> int | None:
    """The smallest count in (lo, hi] whose log-probability is at most
    `limit`; the pmf falls from `lo` (the mode) to `hi`."""
    if log_pmf(hi) > limit:
        return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if log_pmf(mid) <= limit:
            hi = mid
        else:
            lo = mid
    return hi


def _last_at_most(log_pmf, lo: int, hi: int, limit: float) -> int | None:
    """The largest count in [lo, hi) whose log-probability is at most
    `limit`; the pmf rises from `lo` to `hi` (the mode)."""
    if log_pmf(lo) > limit:
        return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if log_pmf(mid) <= limit:
            lo = mid
        else:
            hi = mid
    return lo


def chi_square_p(observed: Sequence[int], expected_shares: Sequence[float]) -> float:
    """The p-value of Pearson's goodness-of-fit test of `observed` counts
    against `expected_shares` (summing to 1). A count in a category expected
    never to occur makes the fit impossible: 0."""
    total = sum(observed)
    if total == 0:
        return 1.0
    statistic, categories = 0.0, 0
    for count, share in zip(observed, expected_shares, strict=True):
        expected = share * total
        if expected <= 0:
            if count:
                return 0.0
            continue
        statistic += (count - expected) ** 2 / expected
        categories += 1
    if categories < 2:
        return 1.0
    return _upper_gamma(categories / 2 - 0.5, statistic / 2)


def total_variation(observed: Sequence[int], expected_shares: Sequence[float]) -> float:
    """Half the summed gap between observed and expected shares: 0 when they
    agree, 1 when they share nothing. The minimum effect frequencies are held
    to."""
    total = sum(observed)
    if total == 0:
        return 0.0
    return 0.5 * sum(
        abs(count / total - share)
        for count, share in zip(observed, expected_shares, strict=True)
    )


def ks_p(distance: float, n: int, m: int) -> float:
    """The asymptotic two-sided p-value of a two-sample Kolmogorov-Smirnov
    `distance` between samples of `n` and `m` values -- the Kolmogorov
    distribution's survival function at sqrt(n*m/(n+m)) * distance."""
    if n <= 0 or m <= 0 or distance <= 0:
        return 1.0
    z = math.sqrt(n * m / (n + m)) * distance
    if z < 0.27:
        return 1.0  # the series converges too slowly here; it is 1 to 1e-9
    total = 0.0
    for j in range(1, 101):
        term = math.exp(-2.0 * j * j * z * z)
        total += term if j % 2 else -term
        if term < 1e-17:
            break
    return min(1.0, max(0.0, 2.0 * total))


def unseen_p(share: float, draws: int) -> float:
    """The chance a value drawn with probability `share` is absent from
    `draws` independent draws: small, and its absence is news."""
    if share <= 0:
        return 1.0
    if share >= 1:
        return 0.0 if draws else 1.0
    return math.exp(draws * math.log1p(-share))


def _upper_gamma(a: float, x: float) -> float:
    """The regularized upper incomplete gamma function Q(a, x): a series
    below a + 1, a continued fraction above (Numerical Recipes, 6.2)."""
    if x <= 0:
        return 1.0
    log_prefix = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1:
        term = total = 1.0 / a
        n = a
        for _ in range(10_000):
            n += 1
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(log_prefix))
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return min(1.0, math.exp(log_prefix) * h)
