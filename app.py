"""Interactive demo: the neural Heston pricer against the exact pricer, live.

  streamlit run app.py"""
import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from neural_heston import domain, heston, model  # noqa: E402
from neural_heston.validate import calibrate, implied_vol, surface_grid  # noqa: E402

st.set_page_config(page_title="Neural Heston pricer", layout="wide")


@st.cache_resource
def net():
    return model.load(os.path.join(os.path.dirname(__file__), "models", "dml.pt"))


@st.cache_data
def batch_speedup(n=10_000):
    """Network against exact pricer on one large batch, the case the network is built for."""
    x = domain.sample(n, seed=41)
    m = net()
    t = time.perf_counter()
    m.predict(x)
    tn = time.perf_counter() - t
    t = time.perf_counter()
    heston.call(x)
    return (time.perf_counter() - t) / tn


def rows(p, k, T):
    k, T = np.broadcast_arrays(np.asarray(k, float), np.asarray(T, float))
    return np.concatenate([np.tile(p, (k.size, 1)), k.reshape(-1, 1), T.reshape(-1, 1)], 1)


st.title("Neural Heston pricer")
st.caption("A neural network trained by differential machine learning to price European options under Heston, "
           "against the exact semi-analytic pricer. Prices are in forward units; strikes as log-moneyness k = ln(K/F).")

B = domain.BOX
with st.sidebar:
    st.header("Heston parameters")
    kappa = st.slider("κ  mean reversion", *B["kappa"], 1.5, 0.1)
    theta = st.slider("θ  long-run variance", *B["theta"], 0.04, 0.005)
    sigma = st.slider("σ  vol of vol", *B["sigma"], 0.5, 0.05)
    rho = st.slider("ρ  spot-vol correlation", *B["rho"], -0.7, 0.05)
    v0 = st.slider("v0  initial variance", *B["v0"], 0.04, 0.005)
    T = st.slider("T  maturity (years)", *B["T"], 0.5, 0.05)
    st.caption(f"Feller ratio 2κθ/σ² = {2 * kappa * theta / sigma ** 2:.2f}"
               + (" (violated: variance can hit zero)" if 2 * kappa * theta < sigma ** 2 else ""))
    st.caption("The sliders are bounded by the training domain: the network is not approved outside it.")

p = np.array([kappa, theta, sigma, rho, v0])
m = net()
tab1, tab2, tab3 = st.tabs(["Smile", "Greeks", "Calibration"])

with tab1:
    k = np.linspace(B["x"][0], B["x"][1], 61) * np.sqrt(T)
    x = rows(p, k, T)
    cn, gn = m.predict(x)
    cr = heston.call(x).numpy()
    ivn, ivr = implied_vol(np.maximum(cn, 0), k, np.full_like(k, T)), implied_vol(cr, k, np.full_like(k, T))
    # implied vol is ill-conditioned where the time value is tiny: shown on the chart, left out of the error
    ok = cr - np.maximum(1 - np.exp(k), 0) > 1e-4
    c1, c2, c3 = st.columns(3)
    c1.metric("Max price error", f"{np.abs(cn - cr).max():.1e}")
    c2.metric("Max implied-vol error", f"{1e4 * np.nanmax(np.abs(ivn - ivr)[ok]):.1f} bp",
              help="Over strikes whose time value exceeds 1e-4 of the forward; below, implied vol is ill-conditioned.")
    c3.metric("Speed-up, batch of 10,000", f"{batch_speedup():.0f}×",
              help="Network forward pass and Greeks against the exact quadrature, on this server's CPU.")
    st.line_chart(pd.DataFrame({"exact": 100 * ivr, "network": 100 * ivn}, index=np.round(k, 3)),
                  x_label="log-moneyness k", y_label="implied vol (%)")
    st.line_chart(pd.DataFrame({"error (bp of implied vol)": 1e4 * (ivn - ivr)}, index=np.round(k, 3)),
                  x_label="log-moneyness k", height=200)

with tab2:
    st.write("Greeks of the network by automatic differentiation, against the exact Greeks of the reference pricer "
             "(itself differentiated by autograd). Derivatives with respect to the seven inputs.")
    _, gr = heston.call_and_grad(x)
    names = ["∂C/∂κ", "∂C/∂θ", "∂C/∂σ", "∂C/∂ρ", "∂C/∂v0 (vega)", "∂C/∂k", "∂C/∂T"]
    pick = st.selectbox("Sensitivity", range(7), index=4, format_func=lambda i: names[i])
    st.line_chart(pd.DataFrame({"exact": gr[:, pick], "network": gn[:, pick]}, index=np.round(k, 3)),
                  x_label="log-moneyness k")

with tab3:
    st.write("A 45-point implied-vol surface (5 maturities × 9 strikes) is generated with the exact pricer at the "
             "sidebar parameters, then calibrated back with the network in the loop, from a fixed starting point.")
    if st.button("Calibrate with the network"):
        ks, Ts = surface_grid()
        target = implied_vol(heston.call(rows(p, ks, Ts)).numpy(), ks, Ts)
        q, fit_bp, sec, nfev = calibrate(lambda z: m.predict(z)[0], ks, Ts, target,
                                         np.array([1.5, 0.05, 0.5, -0.5, 0.05]))
        reprice = 1e4 * (implied_vol(heston.call(rows(q, ks, Ts)).numpy(), ks, Ts) - target)
        c1, c2, c3 = st.columns(3)
        c1.metric("Fit (network)", f"{fit_bp:.1f} bp")
        c2.metric("Reprice with exact pricer", f"{np.sqrt(np.nanmean(reprice ** 2)):.1f} bp")
        c3.metric("Time", f"{sec:.2f} s", help=f"{nfev} surface evaluations")
        st.dataframe(pd.DataFrame({"true": p, "calibrated": q}, index=["κ", "θ", "σ", "ρ", "v0"]).style.format("{:.4f}"))
        st.caption("κ and σ are the least identified parameters on a single surface: different (κ, σ) pairs give "
                   "almost the same smiles. The reprice error is what matters.")

st.caption("Code and validation report: github.com/louisdelacour21/Neural-heston")
