import numpy as np

from src.fivar_model import huber_factor, idio_predict


def decompose_day(covariance, gics, rank=3):
    """Remove daily principal components before applying the industry mask."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    factors = eigenvectors[:, -rank:]
    common = (factors * eigenvalues[-rank:]) @ factors.T
    mask = np.asarray(gics)[:, None] == np.asarray(gics)[None, :]
    specific = (covariance - common) * mask
    return specific, common + specific


def forecast_covariances(gamma, specific, previous_poet):
    """Predict continuous covariance using 251 past days and fixed 22-day bases."""
    if gamma.shape != (251, 200, 200) or specific.shape != gamma.shape:
        raise ValueError("This experiment expects 251 days and 200 assets.")
    _, q_factor = np.linalg.eigh(np.mean(gamma[-22:], axis=0))
    q_factor = q_factor[:, -3:]
    _, q_specific = np.linalg.eigh(np.mean(specific[-22:], axis=0))
    factor_values = np.sum(q_factor[None] * (gamma @ q_factor), axis=1) / 200
    specific_values = np.sum(q_specific[None] * (specific @ q_specific), axis=1)
    factor_prediction = huber_factor(factor_values)
    specific_prediction, diagnostics = idio_predict(
        np.column_stack([factor_values, specific_values])
    )
    coefficients = np.linalg.lstsq(
        np.column_stack([np.ones(250), factor_values[:-1]]),
        factor_values[1:], rcond=None
    )[0]
    ols_prediction = np.r_[1, factor_values[-1]] @ coefficients
    forecasts = {
        "H": (q_factor * (200 * factor_prediction)) @ q_factor.T
        + (q_specific * specific_prediction) @ q_specific.T,
        "OLS": (q_factor * (200 * ols_prediction)) @ q_factor.T
        + np.mean(specific[-22:], axis=0),
        "POET": np.array(previous_poet),
    }
    return forecasts, diagnostics
