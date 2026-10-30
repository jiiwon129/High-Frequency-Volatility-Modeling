import numpy as np


def pdmat(a):
    """Apply the numerical eigenvalue floor used in portfolio optimization."""
    a = (a + a.T) / 2
    vals, v = np.linalg.eigh(a)
    return v * np.maximum(vals, 1e-12) @ v.T


def solve(a, c, initial=None):
    """Minimize variance subject to unit net exposure and a gross-exposure limit."""
    import osqp
    import scipy.sparse as sp
    a = pdmat(a)
    a = a / (np.trace(a) / len(a))
    n = len(a)
    P = sp.block_diag([sp.csc_matrix(2 * a), sp.csc_matrix((n, n))], format='csc')
    I = sp.eye(n, format='csc')
    # Auxiliary variables bound |w| so the gross-exposure constraint stays linear.
    A = sp.vstack([
        sp.csc_matrix(np.r_[np.ones(n), np.zeros(n)][None, :]),
        sp.csc_matrix(np.r_[np.zeros(n), np.ones(n)][None, :]),
        sp.hstack([I, -I]),
        sp.hstack([-I, -I]),
    ], format='csc')
    low = np.r_[1.0, -np.inf, np.full(2 * n, -np.inf)]
    high = np.r_[1.0, c, np.zeros(2 * n)]
    if c == 1:
        P = sp.csc_matrix(2 * a)
        A = sp.vstack([sp.csc_matrix(np.ones((1, n))), I], format='csc')
        low = np.r_[1.0, np.zeros(n)]
        high = np.r_[1.0, np.full(n, np.inf)]
    solver = osqp.OSQP()
    solver.setup(
        P=sp.triu(P, format='csc'), q=np.zeros(P.shape[0]), A=A, l=low, u=high,
        verbose=False, eps_abs=1e-8, eps_rel=1e-8,
        max_iter=100000, polishing=True,
    )
    if initial is not None:
        solver.warm_start(x=initial if c == 1 else np.r_[initial, np.abs(initial)])
    r = solver.solve()
    w = r.x[:n] if r.x is not None else np.full(n, np.nan)
    if r.info.status == 'solved' and abs(w.sum() - 1) < 1e-05 and (np.abs(w).sum() <= c + 1e-05):
        if c == 1:
            w = np.maximum(w, 0)
        w = w / w.sum()
        if c > 1 and np.abs(w).sum() > c:
            mix = max(0.0, (c - 1) / (np.abs(w).sum() - 1))
            w = mix * w + (1 - mix) * np.ones(n) / n
    ok = r.info.status == 'solved' and abs(w.sum() - 1) < 1e-06 and (np.abs(w).sum() <= c + 1e-06)
    if not ok:
        raise RuntimeError(
            f'Optimization failure c={c}: {r.info.status}; '
            f'sum={w.sum()}, gross={abs(w).sum()}'
        )
    return w


def portfolio_covariance(continuous, jump, policy="separate"):
    from src.linalg import project

    continuous = project(continuous, 1e-12)
    if policy == "separate":
        return continuous + project(jump)
    if policy == "combined":
        return project(continuous + jump, 1e-12)
    if policy == "none":
        return continuous
    raise ValueError(f"Unknown jump policy: {policy}")


def realized_variance(returns_10min, weights):
    return float(np.sum((returns_10min @ weights) ** 2))
