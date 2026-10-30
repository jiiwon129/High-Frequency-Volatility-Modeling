import numpy as np

from src.linalg import project


def estimate_day(log_prices):
    """Return continuous PRVM, raw jump covariance, and 10-minute returns."""
    log_prices = np.asarray(log_prices, dtype=float)
    if log_prices.ndim != 2 or log_prices.shape[0] != 390:
        raise ValueError("Expected 390 observations, from 09:30 through 15:59.")
    if not np.isfinite(log_prices).all():
        raise ValueError("Prices contain missing or non-finite values.")

    # The input has no 16:00 tick. Carry the last price forward within the day.
    prices = np.vstack([log_prices, log_prices[-1]])
    returns = np.diff(prices, axis=0)
    m, k = 390, 19
    count = m - k + 1
    grid = np.arange(k + 1) / k
    g = np.minimum(grid, 1 - grid)
    preaveraged = sum(g[j] * returns[j:j + count] for j in range(1, k))

    n_pairs = m - 2 * k + 1
    bipower = m / n_pairs * np.pi / (2 * k) * np.sum(
        abs(preaveraged[:n_pairs]) * abs(preaveraged[k:k + n_pairs]), axis=0
    )
    threshold = 4 * np.sqrt(np.maximum(bipower, 0)) * (k / m) ** 0.47
    keep = abs(preaveraged) <= threshold
    truncated = preaveraged * keep
    noise = np.zeros((log_prices.shape[1], log_prices.shape[1]))
    total_noise = np.zeros_like(noise)
    for j, weight in enumerate(np.diff(g) ** 2):
        window = returns[j:j + count]
        selected = window * keep
        noise += weight * (selected.T @ selected)
        total_noise += weight * (window.T @ window)

    # Appendix A.2 uses integral(g^2) = 1/12, not the discrete weight sum.
    continuous_raw = (truncated.T @ truncated - 0.5 * noise) / (k / 12)
    total_raw = (preaveraged.T @ preaveraged - 0.5 * total_noise) / (k / 12)
    jump = total_raw - continuous_raw
    returns_10min = np.diff(prices[::10], axis=0)
    return project(continuous_raw), jump, returns_10min
