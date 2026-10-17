import numpy as np


def step_1_decompose(gamma_results):
    """Decompose each daily volatility matrix and sort eigenvalues descending."""
    daily_vals = []
    daily_vecs = []

    for gamma_d in gamma_results:
        vals, vecs = np.linalg.eigh(gamma_d)
        idx = np.argsort(vals)[::-1]
        daily_vals.append(vals[idx])
        daily_vecs.append(vecs[:, idx])

    return np.array(daily_vals), np.array(daily_vecs)


def step_2_factor_components(Gamma_history, p, r=3, l=22):
    """Extract factor eigenvectors and factor eigenvalue time series."""
    Gamma_bar = np.mean(Gamma_history[-l:], axis=0)

    vals, vecs = np.linalg.eigh(Gamma_bar)
    idx = np.argsort(vals)[::-1]
    q_F = vecs[:, idx[:r]]

    n_days = len(Gamma_history)
    xi_F = np.zeros((n_days, r))

    for d in range(n_days):
        gamma_d = Gamma_history[d]
        for k in range(r):
            q_k = q_F[:, k]
            val = q_k.T @ gamma_d @ q_k
            xi_F[d, k] = max(val / p, 1e-12)

    return q_F, xi_F


def step_3_4_get_idiosyncratic_matrix(
    Gamma_history,
    xi_F,
    q_F,
    gics_mask,
    p,
):
    """Remove the factor component and apply GICS hard thresholding."""
    n_days = len(Gamma_history)
    Sigma_history = np.zeros_like(Gamma_history)

    for d in range(n_days):
        Gamma_F = p * (q_F @ np.diag(xi_F[d]) @ q_F.T)
        Sigma_raw = Gamma_history[d] - Gamma_F
        Sigma_history[d] = Sigma_raw * gics_mask

    return Sigma_history


def step_5_idio_components(Sigma_history, l=22):
    """Extract idiosyncratic eigenvectors and eigenvalue time series."""
    n_days = len(Sigma_history)
    p = Sigma_history.shape[1]

    Sigma_bar = np.mean(Sigma_history[-l:], axis=0)

    vals, vecs = np.linalg.eigh(Sigma_bar)
    idx = np.argsort(vals)[::-1]
    q_I = vecs[:, idx]

    xi_I = np.zeros((n_days, p))

    for d in range(n_days):
        Sigma_d = Sigma_history[d]
        temp = Sigma_d @ q_I
        xi_I[d] = np.sum(temp * q_I, axis=0)
        xi_I[d] = np.maximum(xi_I[d], 1e-12)

    return q_I, xi_I


def step_8_reconstruct(q_F, pred_next_F, q_I, pred_next_I, p):
    """Reconstruct the next volatility matrix from predicted components."""
    Psi_next = p * (q_F @ np.diag(pred_next_F) @ q_F.T)
    Sigma_next = q_I @ np.diag(pred_next_I) @ q_I.T
    return Psi_next + Sigma_next
