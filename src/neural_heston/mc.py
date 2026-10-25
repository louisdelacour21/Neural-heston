"""Monte Carlo under Heston with Andersen's (2008) quadratic-exponential (QE) scheme: an independent check of the
semi-analytic price (no shared code with heston.py), and later the path generator for the CVA phase."""
import numpy as np


def paths(kappa, theta, sigma, rho, v0, T, steps, n, seed=0, every=None):
    """ln(S_t/F) and v_t on n paths. Returns the terminal (lnS, v), or with every=m the values at every m-th step."""
    rng = np.random.default_rng(seed)
    dt = T / steps
    e = np.exp(-kappa * dt)
    k0 = -rho * kappa * theta * dt / sigma
    k1 = 0.5 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    k2 = 0.5 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    k3 = k4 = 0.5 * dt * (1 - rho * rho)
    x, v = np.zeros(n), np.full(n, float(v0))
    out = []
    for s in range(steps):
        m = theta + (v - theta) * e
        s2 = v * sigma ** 2 * e * (1 - e) / kappa + theta * sigma ** 2 * (1 - e) ** 2 / (2 * kappa)
        psi = s2 / m ** 2
        quad_branch = psi <= 1.5
        vn = np.empty(n)
        z = rng.standard_normal(n)
        ps = psi[quad_branch]
        b2 = 2 / ps - 1 + np.sqrt(2 / ps) * np.sqrt(2 / ps - 1)
        vn[quad_branch] = m[quad_branch] / (1 + b2) * (np.sqrt(b2) + z[quad_branch]) ** 2
        p = (psi[~quad_branch] - 1) / (psi[~quad_branch] + 1)
        beta = (1 - p) / m[~quad_branch]
        uu = rng.random(p.size)
        vn[~quad_branch] = np.where(uu <= p, 0.0, np.log((1 - p) / np.maximum(1 - uu, 1e-300)) / beta)
        x = x + k0 + k1 * v + k2 * vn + np.sqrt(k3 * v + k4 * vn) * rng.standard_normal(n)
        v = vn
        if every and (s + 1) % every == 0:
            out.append((x.copy(), v.copy()))
    return out if every else (x, v)


def call(kappa, theta, sigma, rho, v0, k, T, steps=200, n=400_000, seed=0):
    """Forward call price e^{-rT}E[(S_T - K)^+]/F and its standard error, by antithetic-free QE Monte Carlo."""
    x, _ = paths(kappa, theta, sigma, rho, v0, T, steps, n, seed)
    pay = np.maximum(np.exp(x) - np.exp(k), 0)
    return pay.mean(), pay.std() / np.sqrt(n)
