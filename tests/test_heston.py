"""The reference pricer must be right before anything is learned from it."""
import math

import numpy as np
import pytest
import torch
from riskhub_engine import options

from neural_heston import domain, heston, mc

POINTS = [  # kappa, theta, sigma, rho, v0, k, T: centre, Feller violated, short maturity, deep strikes, near the box edges
    (1.5, 0.04, 0.5, -0.7, 0.04, 0.0, 1.0),
    (0.5, 0.01, 1.0, -0.95, 0.005, 0.1, 0.5),
    (3.0, 0.09, 0.3, -0.3, 0.12, -0.05, 0.05),
    (2.0, 0.06, 0.6, -0.5, 0.03, -0.9, 2.0),
    (5.0, 0.2, 0.1, 0.0, 0.2, 0.6, 1.5),
]


def bs_forward(k, T, vol):
    d1 = (-k + 0.5 * vol * vol * T) / (vol * math.sqrt(T))
    N = lambda z: 0.5 * math.erfc(-z / math.sqrt(2))
    return N(d1) - math.exp(k) * N(d1 - vol * math.sqrt(T))


@pytest.mark.parametrize("p", POINTS)
def test_matches_adaptive_quadrature(p):
    assert heston.call(np.array([p]))[0].item() == pytest.approx(heston.price_quad(*p), abs=1e-10)


def test_matches_quadrature_on_random_domain_points():
    x = domain.sample(64, seed=7)
    ours = heston.call(x).numpy()
    ref = np.array([heston.price_quad(*row) for row in x])
    assert np.max(np.abs(ours - ref)) < 1e-9


@pytest.mark.parametrize("p", POINTS)
def test_matches_riskhub_engine(p):  # two independent implementations already in riskhub-engine (Lewis and P1/P2)
    kappa, theta, sigma, rho, v0, k, T = p
    q = {"kappa": kappa, "theta": theta, "xi": sigma, "rho": rho, "v0": v0}
    ours = heston.call(np.array([p]))[0].item()
    assert options.heston_call(1.0, math.exp(k), T, 0.0, q) == pytest.approx(ours, abs=2e-5)


@pytest.mark.parametrize("p", POINTS[:3])
def test_matches_monte_carlo(p):
    price, se = mc.call(*p, n=300_000, steps=int(200 * p[6]) + 20)
    assert abs(price - heston.call(np.array([p]))[0].item()) < 4 * se + 2e-4


def test_black_scholes_limit():  # sigma -> 0 and v0 = theta: constant variance theta
    for k, T in [(0.0, 1.0), (0.2, 0.5), (-0.3, 2.0)]:
        p = np.array([[2.0, 0.04, 1e-4, -0.5, 0.04, k, T]])
        assert heston.call(p)[0].item() == pytest.approx(bs_forward(k, T, 0.2), abs=1e-5)  # vol-of-vol leaves an O(sigma) skew


def test_autograd_greeks_match_finite_differences():
    x = np.array(POINTS, dtype=float)
    _, g = heston.call_and_grad(x)
    for j in range(7):
        h = 1e-5 * max(1.0, abs(x[:, j]).max())
        up, dn = x.copy(), x.copy()
        up[:, j] += h
        dn[:, j] -= h
        fd = (heston.call(up) - heston.call(dn)).numpy() / (2 * h)
        np.testing.assert_allclose(g[:, j], fd, atol=1e-6, rtol=1e-5, err_msg=heston.NAMES[j])


def test_put_call_parity_and_bounds():
    x = domain.sample(256, seed=3)
    c, p = heston.call(x).numpy(), heston.put(x).numpy()
    K = np.exp(x[:, 5])
    assert np.all(c >= np.maximum(1 - K, 0) - 1e-10) and np.all(c <= 1 + 1e-12)
    assert np.all(p >= np.maximum(K - 1, 0) - 1e-10)


def test_sampler_stays_in_box():
    x = domain.sample(1000, seed=1)
    assert domain.inside(x).all() and x.shape == (1000, 7)
    assert torch.get_default_dtype() == torch.float32  # the pricer must not change torch's global state
