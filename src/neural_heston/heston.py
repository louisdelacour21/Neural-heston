"""Reference Heston (1993) pricer: the "teacher" the network learns from and is validated against.

Prices are in forward units: a European call on F = S e^{(r-q)T} with strike K, divided by the discount factor and by F,
so C(k, T) depends only on the five Heston parameters, the log-moneyness k = ln(K/F) and the maturity T.
The price is the Lewis (2001) single integral

    C = 1 - e^{k/2} / pi * int_0^inf Re[e^{-iuk} phi(u - i/2)] / (u^2 + 1/4) du

with phi the characteristic function of ln(S_T/F) in the "little trap" form (Albrecher et al. 2007), which stays on
the principal branch of the complex log. It is written in torch complex128, so autograd gives exact Greeks and the
differential labels (dC/d inputs) used to train the network; `price_quad` is an independent scipy version for checks."""
import math

import numpy as np
import torch
from scipy.integrate import quad

NAMES = ("kappa", "theta", "sigma", "rho", "v0", "k", "T")  # column order of every input array
NODES = 1024

_x, _w = np.polynomial.legendre.leggauss(NODES)
_X = torch.tensor((_x + 1) / 2, dtype=torch.float64)  # Gauss-Legendre nodes and weights on [0, 1]
_W = torch.tensor(_w / 2, dtype=torch.float64)


def integrated_variance(kappa, theta, v0, T):
    """E[int_0^T v_t dt]: sets how fast the integrand decays, hence where the integral can be cut."""
    return theta * T + (v0 - theta) * (1 - torch.exp(-kappa * T)) / kappa


def cf(z, kappa, theta, sigma, rho, v0, T):
    """E[exp(i z ln(S_T/F))] for complex z (broadcasts)."""
    iz = 1j * z
    a = kappa - rho * sigma * iz
    d = torch.sqrt(a * a + sigma * sigma * (iz + z * z))
    g = (a - d) / (a + d)
    e = torch.exp(-d * T)
    C = kappa * theta / sigma ** 2 * ((a - d) * T - 2 * torch.log((1 - g * e) / (1 - g)))
    D = (a - d) / sigma ** 2 * (1 - e) / (1 - g * e)
    return torch.exp(C + D * v0)


def _call(x):
    kappa, theta, sigma, rho, v0, k, T = (x[:, i:i + 1] for i in range(7))
    with torch.no_grad():  # where the integrand dies out: |phi| ~ exp(-u^2 w / 2) at first, then only ~ exp(-c u)
        w = integrated_variance(kappa, theta, v0, T)
        c = (v0 + kappa * theta * T) * torch.sqrt(1 - rho * rho) / sigma
        U = torch.maximum(torch.sqrt(80 / w), 40 / c).clamp(max=1e5)
        a = 1 / torch.sqrt(w)  # u = a (e^s - 1): fine nodes near 0, coarser in the tail
        S = torch.log1p(U / a)
    s = S * _X
    u = a * torch.expm1(s)
    z = u - 0.5j
    f = (torch.exp(-1j * u * k) * cf(z, kappa, theta, sigma, rho, v0, T)).real / (u * u + 0.25)
    return 1 - torch.exp(k[:, 0] / 2) / math.pi * (S * a)[:, 0] * ((f * torch.exp(s)) @ _W)


def call(x, chunk=2048):
    """Forward call prices for an (n, 7) tensor of inputs in NAMES order; differentiable."""
    x = torch.as_tensor(x, dtype=torch.float64)
    return torch.cat([_call(x[i:i + chunk]) for i in range(0, len(x), chunk)])


def call_and_grad(x, chunk=2048):
    """Prices (n,) and their gradients with respect to the seven inputs (n, 7), as numpy arrays."""
    x = torch.as_tensor(x, dtype=torch.float64)
    prices, grads = [], []
    for i in range(0, len(x), chunk):
        xi = x[i:i + chunk].clone().requires_grad_(True)
        p = _call(xi)
        (g,) = torch.autograd.grad(p.sum(), xi)
        prices.append(p.detach())
        grads.append(g)
    return torch.cat(prices).numpy(), torch.cat(grads).numpy()


def put(x):
    """Forward put prices by put-call parity: P = C - (1 - e^k)."""
    x = torch.as_tensor(x, dtype=torch.float64)
    return call(x) - (1 - torch.exp(x[:, 5]))


def price_quad(kappa, theta, sigma, rho, v0, k, T):
    """Same call price by adaptive scipy quadrature on [0, inf), in plain complex arithmetic: an independent check."""
    def phi(z):
        iz = 1j * z
        a = kappa - rho * sigma * iz
        d = np.sqrt(a * a + sigma * sigma * (iz + z * z))
        g = (a - d) / (a + d)
        e = np.exp(-d * T)
        C = kappa * theta / sigma ** 2 * ((a - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
        D = (a - d) / sigma ** 2 * (1 - e) / (1 - g * e)
        return np.exp(C + D * v0)

    f = lambda u: (np.exp(-1j * u * k) * phi(u - 0.5j)).real / (u * u + 0.25)
    val, _ = quad(f, 0, np.inf, limit=2000, epsabs=1e-13, epsrel=1e-12)
    return 1 - math.exp(k / 2) / math.pi * val
