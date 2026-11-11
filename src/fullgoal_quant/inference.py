"""Post-hoc statistical inference on frozen research results.

Everything here reads already-computed equity curves and fold returns; it never
re-runs a backtest, so it cannot change which strategy is retained.

References
- Politis & Romano (1994), "The Stationary Bootstrap".
- Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier" (PSR).
- Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio" (DSR).
"""
from __future__ import annotations

from math import comb, e, sqrt
from statistics import NormalDist

import numpy as np

ANNUAL = 252
EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


def sharpe(returns, annual=ANNUAL):
    """Annualised Sharpe with rf=0 and ddof=1, matching backtest.metrics."""
    returns = np.asarray(returns, dtype=float)
    std = returns.std(ddof=1) if len(returns) > 1 else 0.0
    return float(returns.mean() / std * sqrt(annual)) if std > 1e-12 else 0.0


def stationary_indices(length, samples, mean_block, rng):
    """Politis-Romano resampling indices: geometric block lengths, circular wrap."""
    if length < 2 or mean_block < 1:
        raise ValueError('需要至少 2 个观测且平均块长 ≥ 1')
    restart = rng.random((samples, length)) < 1.0 / mean_block
    restart[:, 0] = True
    starts = rng.integers(0, length, size=(samples, length))
    position = np.arange(length)
    # Offset since the most recent restart, then step forward from that start.
    last = np.maximum.accumulate(np.where(restart, position, 0), axis=1)
    origin = np.take_along_axis(starts, last, axis=1)
    return (origin + position - last) % length


def _sharpe_rows(matrix, annual=ANNUAL):
    std = matrix.std(axis=1, ddof=1)
    safe = np.where(std > 1e-12, std, 1.0)
    return np.where(std > 1e-12, matrix.mean(axis=1) / safe * sqrt(annual), 0.0)


def bootstrap(returns, benchmark=None, samples=10000, mean_block=10, seed=20261016, level=0.95):
    """Stationary-bootstrap interval for the Sharpe ratio and, if a benchmark is
    given, for the annualised mean daily excess return. Strategy and benchmark
    are resampled with the same indices so their dependence is preserved."""
    returns = np.asarray(returns, dtype=float)
    index = stationary_indices(len(returns), samples, mean_block, np.random.default_rng(seed))
    low, high = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    draws = _sharpe_rows(returns[index])
    result = {'sharpe': sharpe(returns),
              'sharpe_ci': [float(np.percentile(draws, low)), float(np.percentile(draws, high))],
              'prob_sharpe_le_0': float((draws <= 0).mean())}
    if benchmark is not None:
        benchmark = np.asarray(benchmark, dtype=float)
        if len(benchmark) != len(returns):
            raise ValueError('策略与基准的收益序列长度不一致')
        excess = (returns - benchmark)[index].mean(axis=1) * ANNUAL
        result.update(excess_annual=float((returns - benchmark).mean() * ANNUAL),
                      excess_ci=[float(np.percentile(excess, low)), float(np.percentile(excess, high))],
                      prob_excess_le_0=float((excess <= 0).mean()))
    return result


def probabilistic_sharpe(returns, threshold=0.0):
    """PSR: P(true Sharpe > threshold), adjusting for sample length, skewness
    and kurtosis. `threshold` is annualised."""
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    std = returns.std(ddof=1) if n > 1 else 0.0
    if n < 3 or std <= 1e-12:
        return 0.0
    observed = returns.mean() / std
    centred = returns - returns.mean()
    population = returns.std(ddof=0)
    skew = float((centred ** 3).mean() / population ** 3)
    kurtosis = float((centred ** 4).mean() / population ** 4)
    variance = 1 - skew * observed + (kurtosis - 1) / 4 * observed ** 2
    if variance <= 0:
        return 0.0
    return _NORMAL.cdf((observed - threshold / sqrt(ANNUAL)) * sqrt(n - 1) / sqrt(variance))


def expected_max_sharpe(trial_sharpes, trials=None):
    """Expected maximum annualised Sharpe among `trials` unskilled strategies
    whose Sharpe estimates have the cross-sectional variance observed."""
    trial_sharpes = np.asarray(trial_sharpes, dtype=float)
    trials = float(trials if trials is not None else len(trial_sharpes))
    if len(trial_sharpes) < 2 or trials <= 1:
        return 0.0
    spread = trial_sharpes.std(ddof=1)
    return float(spread * ((1 - EULER_GAMMA) * _NORMAL.inv_cdf(1 - 1 / trials)
                           + EULER_GAMMA * _NORMAL.inv_cdf(1 - 1 / (trials * e))))


def effective_trials(return_matrix):
    """Trials discounted for correlation: rho + (1 - rho) * N, where rho is the
    mean pairwise correlation of the trials' daily returns."""
    matrix = np.asarray(return_matrix, dtype=float)
    count = matrix.shape[0]
    if count < 2:
        return float(count)
    correlation = np.corrcoef(matrix)
    mean = float(correlation[np.triu_indices(count, 1)].mean())
    mean = min(max(mean, 0.0), 1.0)
    return mean + (1 - mean) * count


def sign_test(values):
    """One-sided exact binomial p-value for 'more positive periods than chance'."""
    values = [v for v in values if v != 0]
    n, k = len(values), sum(v > 0 for v in values)
    p = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else 1.0
    return {'positive': k, 'periods': n, 'p_value': float(p)}
