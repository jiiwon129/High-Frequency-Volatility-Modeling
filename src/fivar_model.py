import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Lasso


def huber_factor(x):
    """Forecast three common-factor eigenvalues from a 251-day window."""
    scale = np.sqrt(np.mean(x * x))
    z = x / max(scale, 1e-15)
    a = (251 / np.log(200)) ** 0.25
    X = np.column_stack([np.ones(250), np.clip(z[:-1], -4 * a, 4 * a)])
    last = np.r_[1, z[-1]]
    tau = 0.25 * a
    predictions = []
    for col in range(3):
        y = z[1:, col]

        def fun(b):
            r = X @ b - y
            ar = abs(r)
            loss = np.mean(np.where(ar <= tau, 0.5 * r * r, tau * (ar - 0.5 * tau)))
            gradient = X.T @ np.clip(r, -tau, tau) / len(r)
            return loss, gradient

        fit = minimize(
            fun, np.linalg.lstsq(X, y, rcond=None)[0], jac=True,
            method='L-BFGS-B',
            options={'gtol': 1e-9, 'ftol': 1e-13, 'maxiter': 2000},
        )
        if not fit.success and np.max(abs(fun(fit.x)[1])) > 1e-06:
            raise RuntimeError(f'factor convergence {fit.message}')
        predictions.append(last @ fit.x * scale)
    return np.array(predictions)


def idio_predict(x):
    """Select the regularization strength by response-wise BIC."""
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    sd = np.maximum(sd, 1e-14)
    z = (x - mu) / sd
    tau = 4 * (251 / np.log(200)) ** 0.25
    X = np.asfortranarray(np.column_stack([np.ones(250), np.clip(z[:-1], -tau, tau)]))
    Y = np.asfortranarray(z[1:, 3:])
    N = len(X)
    last = np.r_[1, z[-1]]
    best = np.full(200, np.inf)
    pred = np.zeros(200)
    selected = np.zeros(200)
    dfs = np.zeros(200)
    model = Lasso(fit_intercept=False, max_iter=20000, tol=1e-07, warm_start=True, precompute=True)
    robust_fits = 0
    maxkkt = 0.0
    for c in np.geomspace(10, 0.1, 21):
        eta = c * np.sqrt(np.log(200) / 251)
        model.set_params(alpha=eta)
        model.fit(X, Y)
        coef = model.coef_.copy()
        residual = Y - X @ coef.T
        initial_grad = -(X.T @ np.clip(residual, -tau, tau) / N).T
        initial_kkt = np.where(
            abs(coef) > 1e-8,
            abs(initial_grad + eta * np.sign(coef)),
            np.maximum(abs(initial_grad) - eta, 0),
        ).max(axis=1)
        # Refine responses affected by Huber clipping or incomplete convergence.
        for j in np.flatnonzero(np.any(abs(residual) > tau, axis=0) | (initial_kkt > 1e-05)):
            robust_fits += 1
            b = coef[j].copy()
            irls = Lasso(fit_intercept=False, max_iter=100000, tol=1e-12, warm_start=True)
            for iteration in range(2000):
                r = Y[:, j] - X @ b
                weights = np.minimum(1, tau / np.maximum(abs(r), 1e-15))
                irls.set_params(alpha=eta * N / weights.sum())
                irls.fit(X, Y[:, j], sample_weight=weights)
                new = irls.coef_
                gradient = -X.T @ np.clip(Y[:, j] - X @ new, -tau, tau) / N
                kk = np.where(
                    abs(new) > 1e-8,
                    abs(gradient + eta * np.sign(new)),
                    np.maximum(abs(gradient) - eta, 0),
                ).max()
                if kk < 1e-06:
                    b = new
                    break
                b = new
            coef[j] = b
        residual = Y - X @ coef.T
        grad = -(X.T @ np.clip(residual, -tau, tau) / N).T
        active = abs(coef) > 1e-08
        kkt = np.where(active, abs(grad + eta * np.sign(coef)), np.maximum(abs(grad) - eta, 0)).max()
        maxkkt = max(maxkkt, float(kkt))
        if kkt > 2e-05:
            raise RuntimeError(f'Huber LASSO did not converge: KKT {kkt}')
        df = (abs(coef) > 1e-08).sum(axis=1)
        bic = N * np.log(np.maximum(np.mean(residual ** 2, axis=0), 1e-30)) + df * np.log(N)
        better = bic < best
        best[better] = bic[better]
        pred[better] = (coef @ last)[better]
        selected[better] = c
        dfs[better] = (abs(coef[:, 1:]) > 1e-08).sum(axis=1)[better]
    diagnostics = {
        'c_eta_mean': float(selected.mean()),
        'c_eta_min': float(selected.min()),
        'c_eta_max': float(selected.max()),
        'nonzero_mean': float(dfs.mean()),
        'robust_refits': robust_fits,
        'max_kkt': maxkkt,
    }
    return pred * sd[3:] + mu[3:], diagnostics
