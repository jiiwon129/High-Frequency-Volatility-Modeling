import numpy as np
from joblib import Parallel, delayed


def calculate_prvm_strict(
    price_data,
    p=200,
    m=390,
    num_days=997,
    c0=4,
    alpha_u=0.47,
):
    """Estimate daily volatility matrices with pre-averaging, jump truncation,
    noise correction, and PSD projection.

    Extracted from the final research notebook and cleaned for portfolio use.
    """
    if hasattr(price_data, "values"):
        price_data = price_data.values

    K = int(np.floor(m**0.5))

    l_vec = np.arange(1, K + 1)
    g_func = np.minimum(l_vec / K, 1 - l_vec / K)

    w_bar = g_func[:-1].reshape(-1, 1)

    g_padded = np.concatenate(([0], g_func))
    w_hat = (g_padded[1:] - g_padded[:-1]) ** 2
    w_hat = w_hat.reshape(-1, 1)

    psi_K = np.sum(g_func[:-1] ** 2)

    def process_day(d):
        day_price = price_data[d * m : (d + 1) * m + 1, :]
        raw_diffs = day_price[1:] - day_price[:-1]

        current_m = len(raw_diffs)
        T_effective = current_m - K + 1

        if T_effective <= 0:
            return np.zeros((p, p))

        gamma_mat = np.zeros((p, p))

        Y_bar = np.zeros((T_effective, p))
        for k in range(T_effective):
            window_returns = raw_diffs[k : k + K - 1]
            Y_bar[k] = np.sum(window_returns * w_bar, axis=0)

        abs_Y = np.abs(Y_bar)
        const_pi = (m / (m - 2 * K + 1)) * (np.pi / (2 * K))

        lim_idx = min(current_m - 2 * K + 1, T_effective - K)
        if lim_idx > 0:
            T_hat_val = const_pi * np.sum(
                abs_Y[:lim_idx] * abs_Y[K : K + lim_idx],
                axis=0,
            )
        else:
            T_hat_val = np.zeros(p)

        u_im = c0 * np.sqrt(T_hat_val) * (K / m) ** alpha_u
        is_continuous = abs_Y <= u_im[None, :]

        for k in range(T_effective):
            y_k = Y_bar[k]
            ind_k = is_continuous[k]

            if not np.any(ind_k):
                continue

            window_returns_full = raw_diffs[k : k + K]
            weighted_noise_rets = window_returns_full * w_hat
            Y_hat_local = weighted_noise_rets.T @ window_returns_full

            Y_bar_outer = np.outer(y_k, y_k)
            term = Y_bar_outer - 0.5 * Y_hat_local

            mask = np.outer(ind_k, ind_k)
            gamma_mat += term * mask

        gamma_final = gamma_mat / psi_K
        eigvals, eigvecs = np.linalg.eigh(gamma_final)
        eigvals_psd = np.maximum(eigvals, 1e-8)

        return eigvecs @ np.diag(eigvals_psd) @ eigvecs.T

    results = Parallel(n_jobs=-1)(
        delayed(process_day)(d) for d in range(num_days)
    )

    return np.array(results)
