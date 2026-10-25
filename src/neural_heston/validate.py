"""Independent validation of a trained pricer, along the six axes of the report:

  1. accuracy in the domain, by risk-relevant slices, and outside it;
  2. static no-arbitrage on a grid the network never saw (strike monotonicity, butterflies, calendar spreads);
  3. Greeks by automatic differentiation of the network against the exact Greeks of the reference;
  4. calibration: recover known parameters from a synthetic surface with the network, compare with the reference;
  5. speed and training cost;
  6. what the results imply for use (limits, guardrails), written in the report.

  python -m neural_heston.validate models/dml.pt [models/plain.pt ...]   # writes report/results.json and figures"""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.optimize import least_squares

from . import data, domain, heston, model

REPORT = os.path.join(os.path.dirname(__file__), "..", "..", "report")
P5 = slice(0, 5)


# ---------- helpers ----------
def implied_vol(c, k, T):
    """Black implied volatility of forward call prices, by bisection (vectorised); nan below intrinsic + 1e-12."""
    from scipy.special import ndtr
    lo, hi = np.full_like(c, 1e-4), np.full_like(c, 5.0)
    sT = np.sqrt(T)
    for _ in range(80):
        mid = (lo + hi) / 2
        d1 = (-k + 0.5 * mid ** 2 * T) / (mid * sT)
        bs = ndtr(d1) - np.exp(k) * ndtr(d1 - mid * sT)
        lo, hi = np.where(bs < c, mid, lo), np.where(bs < c, hi, mid)
    iv = (lo + hi) / 2
    iv[c <= np.maximum(1 - np.exp(k), 0) + 1e-12] = np.nan
    return iv


def stats(err):
    err = err[np.isfinite(err)]
    return {"n": int(err.size), "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "p99": float(np.quantile(np.abs(err), 0.99)), "max": float(np.abs(err).max())}


def second_k(f, x):
    """C, dC/dk and d2C/dk2 of a torch pricer f at the rows of x."""
    x = torch.as_tensor(x).clone().requires_grad_(True)
    p = f(x)
    (g,) = torch.autograd.grad(p.sum(), x, create_graph=True)
    (h,) = torch.autograd.grad(g[:, 5].sum(), x)
    return p.detach().numpy().astype(float), g.detach().numpy().astype(float), h[:, 5].numpy().astype(float)


def net_fn(m):
    return lambda x: m(x.float())


def ref_fn(x):
    return heston.call(x.double())


# ---------- 1. accuracy ----------
def slices(x):
    sT = np.sqrt(x[:, 6])
    u = np.stack([(x[:, 0] - 0.5) / 4.5, (x[:, 1] - 0.01) / 0.19, (x[:, 2] - 0.1) / 0.9, (x[:, 3] + 0.95) / 0.95,
                  (x[:, 4] - 0.005) / 0.195, (x[:, 5] / sT + 0.8) / 1.3, (sT - np.sqrt(0.05)) / (np.sqrt(2) - np.sqrt(0.05))], 1)
    return {
        "all": np.ones(len(x), bool),
        "T < 0.15": x[:, 6] < 0.15,
        "T > 1": x[:, 6] > 1,
        "Feller violated": domain.feller(x) < 1,
        "Feller satisfied": domain.feller(x) >= 1,
        "rho < -0.85": x[:, 3] < -0.85,
        "sigma > 0.8": x[:, 2] > 0.8,
        "deep OTM call (x > 0.3)": x[:, 5] / sT > 0.3,
        "near box edge (any input within 5%)": ((u < 0.05) | (u > 0.95)).any(1),
    }


OUT_BOXES = {  # each widens one side of the box; the network is used outside what it was trained on
    "T in [0.01, 0.05]": {"T": (0.01, 0.05)},
    "T in [2, 3]": {"T": (2.0, 3.0)},
    "rho in [-0.99, -0.95]": {"rho": (-0.99, -0.95)},
    "rho in [0, 0.3]": {"rho": (0.0, 0.3)},
    "sigma in [1, 1.5]": {"sigma": (1.0, 1.5)},
    "kappa in [5, 8]": {"kappa": (5.0, 8.0)},
    "v0 in [0.2, 0.4]": {"v0": (0.2, 0.4)},
    "x in [0.5, 0.8]": {"x": (0.5, 0.8)},
}


def accuracy(m, te):
    p, g = m.predict(te["x"])
    x = te["x"]
    iv_ref = implied_vol(te["price"], x[:, 5], x[:, 6])
    iv_net = implied_vol(np.maximum(p, 0), x[:, 5], x[:, 6])
    # implied vol is ill-conditioned where the time value is near zero: deep OTM, but also deep ITM calls
    vega_ok = te["price"] - np.maximum(1 - np.exp(x[:, 5]), 0) > 1e-4
    out = {"in_domain": {}, "out_of_domain": {}}
    for name, mask in slices(x).items():
        out["in_domain"][name] = {"price": stats((p - te["price"])[mask]),
                                  "iv_bp": stats(1e4 * (iv_net - iv_ref)[mask & vega_ok])}
    for name, ch in OUT_BOXES.items():
        box = dict(domain.BOX, **ch)
        d = data.build(2048, seed=11, box=box)
        q, _ = m.predict(d["x"])
        out["out_of_domain"][name] = {"price": stats(q - d["price"])}
    return out, {"x": x, "err": p - te["price"], "iv_err_bp": 1e4 * (iv_net - iv_ref)}


# ---------- 2. no-arbitrage ----------
def arbitrage(m, n_sets=400, seed=21):
    """Strike and maturity grids for random parameter sets, never seen in training (other Sobol seed, regular grid)."""
    params = domain.sample(n_sets, seed=seed)[:, P5]
    xs = np.linspace(-0.8, 0.5, 45)               # strikes per maturity, as k / sqrt(T) across the box
    Ts = np.linspace(0.05, 2.0, 25)
    kc = np.linspace(-0.3, 0.2, 11)               # fixed strikes for calendar spreads, kept where they are in the box
    P, X, T = np.meshgrid(np.arange(n_sets), xs, Ts, indexing="ij")
    xb = np.concatenate([params[P.ravel()], (X * np.sqrt(T)).ravel()[:, None], T.ravel()[:, None]], 1)
    P, K, T = np.meshgrid(np.arange(n_sets), kc, Ts, indexing="ij")
    xc = np.concatenate([params[P.ravel()], K.ravel()[:, None], T.ravel()[:, None]], 1)
    ok = domain.inside(xc).reshape(n_sets, len(kc), len(Ts))
    ok = ok[:, :, 1:] & ok[:, :, :-1]
    res = {}
    for name, f in [("network", lambda z: m.predict(z)[0]), ("reference", lambda z: heston.call(z).numpy())]:
        c = f(xb).reshape(n_sets, len(xs), len(Ts))
        Kv = np.exp(xs[None, :, None] * np.sqrt(Ts)[None, None, :])
        dK = np.diff(c, axis=1) / np.diff(Kv, axis=1)            # call spread: -1 <= dC/dK <= 0
        slope = np.diff(dK, axis=1)                             # butterfly: dC/dK non-decreasing in K
        cal = np.diff(f(xc).reshape(n_sets, len(kc), len(Ts)), axis=2)[ok]  # calendar at fixed K/F: non-decreasing in T
        res[name] = {
            "points": int(c.size + ok.size),
            "call_spread": float(np.mean((dK > 1e-9) | (dK < -1 - 1e-9))),
            "butterfly": float(np.mean(slope < -1e-9)),
            "calendar": float(np.mean(cal < -1e-9)),
            "butterfly_worst": float(max(0.0, -slope.min())),
            "calendar_worst": float(max(0.0, -cal.min())),
        }
    return res


# ---------- 3. Greeks ----------
GREEKS = ["delta", "gamma", "vega_v0", "dC/dtheta", "dC/dkappa", "dC/dsigma", "dC/drho", "dC/dT"]


def greeks_of(f, x):
    p, g, h = second_k(f, x)
    return np.stack([p - g[:, 5],      # dV/dF at F = 1: V = F C(ln K - ln F)
                     h - g[:, 5],      # d2V/dF2 at F = 1
                     g[:, 4], g[:, 1], g[:, 0], g[:, 2], g[:, 3], g[:, 6]], 1)


def greeks(m, te, n=4096):
    x = te["x"][:n]
    gn = greeks_of(net_fn(m), torch.tensor(x, dtype=torch.float32))
    gr = greeks_of(ref_fn, torch.tensor(x, dtype=torch.float64))
    out = {}
    for i, name in enumerate(GREEKS):
        e = gn[:, i] - gr[:, i]
        out[name] = {"rmse": float(np.sqrt(np.mean(e ** 2))), "scale": float(np.std(gr[:, i])),
                     "rel_rmse": float(np.sqrt(np.mean(e ** 2)) / np.std(gr[:, i])),
                     "p99": float(np.quantile(np.abs(e), 0.99))}
    return out, {"net": gn, "ref": gr}


# ---------- 4. calibration ----------
def surface_grid():
    Ts = np.array([0.1, 0.25, 0.5, 1.0, 1.5])
    xs = np.linspace(-0.6, 0.4, 9)
    T, X = np.meshgrid(Ts, xs, indexing="ij")
    return (X * np.sqrt(T)).ravel(), T.ravel()


def calibrate(pricer, ks, Ts, target_iv, start):
    """Least squares in implied vol, parameters mapped to the box by a logistic so the fit stays in the domain."""
    lo = np.array([domain.BOX[k][0] for k in ("kappa", "theta", "sigma", "rho", "v0")])
    hi = np.array([domain.BOX[k][1] for k in ("kappa", "theta", "sigma", "rho", "v0")])
    to_p = lambda z: lo + (hi - lo) / (1 + np.exp(-z))
    z0 = -np.log((hi - lo) / (np.clip(start, lo + 1e-6, hi - 1e-6) - lo) - 1)

    def resid(z):
        x = np.concatenate([np.tile(to_p(z), (len(ks), 1)), ks[:, None], Ts[:, None]], 1)
        return 1e4 * (implied_vol(np.maximum(pricer(x), 1e-14), ks, Ts) - target_iv)

    t = time.time()
    # diff_step suits a float32 network: the default steps (~1e-8) are below its rounding and give a zero Jacobian
    r = least_squares(lambda z: np.nan_to_num(resid(z), nan=1e3), z0, method="trf", max_nfev=400, diff_step=1e-3)
    return to_p(r.x), float(np.sqrt(np.mean(r.fun ** 2))), time.time() - t, r.nfev


def calibration(m, n=25, seed=31):
    ks, Ts = surface_grid()
    truth = domain.sample(n, seed=seed)[:, P5]
    start = np.array([1.5, 0.05, 0.5, -0.5, 0.05])
    rows = []
    for p in truth:
        x = np.concatenate([np.tile(p, (len(ks), 1)), ks[:, None], Ts[:, None]], 1)
        iv = implied_vol(heston.call(x).numpy(), ks, Ts)
        row = {"truth": p.tolist()}
        for name, f in [("network", lambda z: m.predict(z)[0]), ("reference", lambda z: heston.call(z).numpy())]:
            q, fit_bp, sec, nfev = calibrate(f, ks, Ts, iv, start)
            # how well the calibrated parameters reprice the surface, with the exact pricer
            xq = np.concatenate([np.tile(q, (len(ks), 1)), ks[:, None], Ts[:, None]], 1)
            reprice = 1e4 * (implied_vol(heston.call(xq).numpy(), ks, Ts) - iv)
            row[name] = {"params": q.tolist(), "fit_bp": fit_bp, "reprice_bp": float(np.sqrt(np.nanmean(reprice ** 2))),
                         "seconds": sec, "evals": nfev}
        rows.append(row)
    return rows


# ---------- 5. speed ----------
def speed(m, sizes=(1, 100, 10_000)):
    out = {}
    for n in sizes:
        x = domain.sample(n, seed=41)
        reps = max(1, 2000 // n)
        t = time.perf_counter()
        for _ in range(reps):
            with torch.no_grad():
                m(torch.tensor(x, dtype=torch.float32))
        tn = (time.perf_counter() - t) / reps
        t = time.perf_counter()
        for _ in range(max(1, reps // 10)):
            heston.call(x)
        tr = (time.perf_counter() - t) / max(1, reps // 10)
        out[str(n)] = {"network_s": tn, "reference_s": tr, "speedup": tr / tn}
    return out


def main(argv=None):
    paths = (argv if argv is not None else sys.argv[1:]) or ["models/dml.pt"]
    os.makedirs(REPORT, exist_ok=True)
    te = data.load("test")
    results, raw = {}, {}
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        m = model.load(path)
        print(f"== {name}", flush=True)
        r = {"meta": m.meta}
        r["accuracy"], raw_acc = accuracy(m, te)
        print("  accuracy done", flush=True)
        r["arbitrage"] = arbitrage(m)
        print("  arbitrage done", flush=True)
        r["greeks"], raw_g = greeks(m, te)
        print("  greeks done", flush=True)
        if name == "dml":
            r["calibration"] = calibration(m)
            print("  calibration done", flush=True)
            r["speed"] = speed(m)
        results[name] = r
        raw[name] = {**raw_acc, "greeks_net": raw_g["net"], "greeks_ref": raw_g["ref"]}
    with open(os.path.join(REPORT, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    np.savez_compressed(os.path.join(REPORT, "raw.npz"),
                        **{f"{n}__{k}": v for n, d in raw.items() for k, v in d.items()})
    print("written report/results.json")


if __name__ == "__main__":
    sys.exit(main())
