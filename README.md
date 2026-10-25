# Neural Heston

A neural-network pricer for European options under the Heston model, trained by **differential machine learning**
(Huge & Savine, 2020), and an **independent validation report** written the way a model-validation team would.

The network replaces the semi-analytic Heston integral by an MLP written as a Black price plus a learned correction,
with Greeks by automatic differentiation. The point of the project is not only the speed-up but the evidence that the
approximation can be trusted, and where it cannot.

**[Validation report (PDF)](report/validation_report.pdf)** · **[Live demo](https://neural-heston.streamlit.app/)** (or locally: `streamlit run app.py`)

## Results

On 16,384 test points never seen in training, inside the training box:

| | price-only network | differential network |
| --- | --: | --: |
| price RMSE (forward units) | 3.8e-5 | **9.6e-6** |
| implied-vol error, mean / 99th pct. | 2.6 / 24.9 bp | **0.8 / 7.5 bp** |
| gamma error (RMSE / dispersion) | 3.3 % | **0.7 %** |
| butterfly violations on a 555k-point grid | 0.23 % | **0.08 %** |
| calibration, reprice error with the exact pricer (median / worst of 25) | | **0.4 / 9.7 bp** |

Opinion of the report: **fit for purpose inside the training box**, with prices floored at intrinsic value (0.13 % of
raw outputs are slightly negative in the far wings) and no use outside the box, where errors grow up to tenfold. The
speed-up against the vectorised reference is 26× on a batch of 10,000 prices on CPU: the network pays off on large
batches such as exposure simulation (CVA), the next step of the project.

## What is in the repository

| Module | Role |
| --- | --- |
| `heston.py` | Reference pricer: Lewis (2001) integral, "little trap" characteristic function, in torch complex128 so autograd gives exact Greeks |
| `mc.py` | Andersen (2008) QE Monte Carlo, an independent check of the reference |
| `domain.py` | Training box (κ, θ, σ, ρ, v0, strike, maturity) and Sobol samplers |
| `data.py` | Training / test sets: prices **and** their gradients with respect to the seven inputs |
| `model.py` | Twin network, trained on prices and derivatives |
| `validate.py` | Six-axis validation: accuracy, no-arbitrage, Greeks, calibration, speed, limits |
| `figures.py` | Report figures |
| `report.py` | Fills `report/report.md` from `results.json`, renders it to HTML and PDF (headless Edge or Chrome) |
| `app.py` | Streamlit demo: smile, Greeks and calibration, network against the exact pricer |

## Reproduce

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install -e .[dev]
python -m pytest                       # reference pricer checks
python -m neural_heston.data           # ~6 min on CPU
python -m neural_heston.train --width 256 --depth 5 --epochs 250                      # differential ML, ~70 min
python -m neural_heston.train --width 256 --depth 5 --epochs 250 --lam 0 --name plain  # price-only baseline
python -m neural_heston.validate models/v0_plain.pt models/v0_dml.pt models/plain.pt models/dml.pt
python -m neural_heston.figures
python -m neural_heston.report         # report/validation_report.pdf
streamlit run app.py
```

Everything runs on a laptop CPU; no GPU or paid service is needed. The trained models are in `models/`; the two
`v0_*` files are the first version (no control variate, 4 × 128), kept for the ablation of the report.

## Conventions

Prices are in forward units, C / (DF · F), with k = ln(K/F). A dollar price is DF · F · C, so the network does not
depend on rates or dividends. Puts follow by parity.

## References

- Heston (1993); Lewis (2001); Albrecher, Mayer, Schoutens & Tistaert (2007), *The little Heston trap*.
- Andersen (2008), *Simple and efficient simulation of the Heston stochastic volatility model*.
- Huge & Savine (2020), *Differential machine learning*.
- Horvath, Muguruza & Tomas (2021), *Deep learning volatility*.
- Sridi & Bilokon (2023), arXiv 2309.07843.
