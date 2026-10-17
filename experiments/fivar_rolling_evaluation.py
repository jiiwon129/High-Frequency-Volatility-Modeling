import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


def get_daily_poet_target_final(gamma_data, gics_data, n_win=251, l_win=22):
    """Construct daily factor/idiosyncratic targets with GICS filtering."""
    if hasattr(gics_data, "values"):
        gics_data = gics_data.values.flatten()

    gics_mask = (
        gics_data[:, None] == gics_data[None, :]
    ).astype(int)

    num_days = len(gamma_data)
    r = 3

    def process_day(d):
        if d < n_win:
            return None

        target_win = gamma_data[d - l_win + 1 : d + 1]
        _, vecs = np.linalg.eigh(np.mean(target_win, axis=0))
        q_F = vecs[:, -r:]

        gamma_d = gamma_data[d]
        xi_Fd = np.array([
            max(q_F[:, k].T @ gamma_d @ q_F[:, k], 1e-12)
            for k in range(r)
        ])

        gamma_F = q_F @ np.diag(xi_Fd) @ q_F.T
        gamma_I = (gamma_d - gamma_F) * gics_mask
        return gamma_F + gamma_I

    results = Parallel(n_jobs=-1)(
        delayed(process_day)(d) for d in range(num_days)
    )
    return np.array([x for x in results if x is not None])


class FIVARFinalStrict:
    """Rolling factor/idiosyncratic volatility prediction experiment."""

    def __init__(self, p, r=3, l=22, n=251):
        self.p = p
        self.r = r
        self.l = l
        self.n = n
        self.cF1 = 4.0
        self.cF2 = 0.25
        self.cI1 = 4.0
        self.cI2 = 4.0
        self.c_eta = 1.0

    def run_one_day(self, history, target, gics_mask):
        Gamma_bar = np.mean(history[-self.l :], axis=0)
        _, vecs = np.linalg.eigh(Gamma_bar)
        q_F = vecs[:, -self.r :]

        xi_F = np.array([
            [
                max(
                    q_F[:, k].T @ history[i] @ q_F[:, k] / self.p,
                    1e-12,
                )
                for k in range(self.r)
            ]
            for i in range(len(history))
        ])

        Sigma_hist = np.zeros_like(history)
        for i in range(len(history)):
            factor = self.p * (
                q_F @ np.diag(xi_F[i]) @ q_F.T
            )
            Sigma_hist[i] = (history[i] - factor) * gics_mask

        Sigma_bar = np.mean(Sigma_hist[-self.l :], axis=0)
        _, q_I = np.linalg.eigh(Sigma_bar)

        xi_I = np.array([
            np.maximum(
                np.diag(q_I.T @ Sigma_hist[i] @ q_I),
                1e-12,
            )
            for i in range(len(history))
        ])

        pred_F = self._predict(xi_F, role="factor")
        pred_I = self._predict(xi_I, role="idio")

        pred_Gamma = (
            self.p * (q_F @ np.diag(pred_F) @ q_F.T)
            + q_I @ np.diag(pred_I) @ q_I.T
        )

        return np.sum((pred_Gamma - target) ** 2)

    def _predict(self, xi, role):
        X = xi[:-1]
        y = xi[1:]

        scaling = (self.n / np.log(self.p)) ** 0.25
        tau = (
            max(0.25 * scaling, 1.0)
            if role == "factor"
            else max(4.0 * scaling, 1.35)
        )

        scaler = StandardScaler()

        if role == "factor":
            sigma_F = np.sqrt(
                np.sum(xi**2) / (self.n * self.r)
            )
            limit_x = self.cF1 * sigma_F * scaling
            limit_y = self.cF2 * sigma_F * scaling

            X_scaled = np.clip(
                scaler.fit_transform(X),
                -limit_x,
                limit_x,
            )
            y_scaled = np.clip(y, -limit_y, limit_y)

            last = np.clip(
                scaler.transform(xi[-1].reshape(1, -1)),
                -limit_x,
                limit_x,
            )

            preds = []
            for i in range(self.r):
                model = SGDRegressor(
                    loss="huber",
                    epsilon=tau,
                )
                model.fit(X_scaled, y_scaled[:, i])
                preds.append(model.predict(last)[0])

            return np.array(preds)

        limit = self.cI1 * scaling
        eta_I = self.c_eta * np.sqrt(
            np.log(self.p) / self.n
        )

        X_scaled = np.clip(
            scaler.fit_transform(
                X.flatten().reshape(-1, 1)
            ),
            -limit,
            limit,
        )
        y_scaled = np.clip(y.flatten(), -limit, limit)

        y_mean = np.mean(y_scaled)
        y_std = np.std(y_scaled) + 1e-9

        model = SGDRegressor(
            loss="huber",
            epsilon=tau,
            penalty="l1",
            alpha=eta_I,
        )
        model.fit(
            X_scaled,
            (y_scaled - y_mean) / y_std,
        )

        last = np.clip(
            scaler.transform(
                xi[-1].flatten().reshape(-1, 1)
            ),
            -limit,
            limit,
        )

        return model.predict(last) * y_std + y_mean


def execute_fivar_rolling_evaluation(gamma_results, gics):
    """Run the 2018–2019 rolling MSPE experiment from the final notebook."""
    targets_poet = get_daily_poet_target_final(
        gamma_results,
        gics,
    )

    gics_values = gics.values.flatten()
    gics_mask = (
        gics_values[:, None] == gics_values[None, :]
    ).astype(int)

    dates_full = pd.date_range(
        "2016-01-04",
        periods=len(targets_poet) + 251,
        freq="B",
    )[251:]

    eval_idx = np.where(
        (dates_full.year >= 2018)
        & (dates_full.year <= 2019)
    )[0]

    model = FIVARFinalStrict(
        p=gamma_results.shape[2]
    )

    results = Parallel(
        n_jobs=-1,
        verbose=1,
    )(
        delayed(model.run_one_day)(
            targets_poet[t - 251 : t],
            targets_poet[t],
            gics_mask,
        )
        for t in eval_idx
    )

    return pd.Series(
        results,
        index=dates_full[eval_idx],
    )
