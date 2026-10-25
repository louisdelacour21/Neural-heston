"""Training domain: the box of inputs the network is trained on and certified for, and samplers over it.

The strike is sampled as a standardised moneyness x = k / sqrt(T), so that short maturities get strikes close to the
forward (where their price lives) and long maturities wider ones. Maturities are uniform in sqrt(T) to put more points
on the short end, where the price surface curves most."""
import numpy as np
from scipy.stats import qmc

from .heston import NAMES

BOX = {
    "kappa": (0.5, 5.0),
    "theta": (0.01, 0.2),
    "sigma": (0.1, 1.0),
    "rho": (-0.95, 0.0),
    "v0": (0.005, 0.2),
    "x": (-0.8, 0.5),   # k / sqrt(T), k = ln(K/F)
    "T": (0.05, 2.0),
}
KEYS = ("kappa", "theta", "sigma", "rho", "v0", "x", "T")


def from_unit(u, box=BOX):
    """Maps points of [0, 1]^7 to inputs (n, 7) in heston.NAMES order."""
    lo = np.array([box[k][0] for k in KEYS])
    hi = np.array([box[k][1] for k in KEYS])
    z = lo + u * (hi - lo)
    sT = np.sqrt(box["T"][0]) + u[:, 6] * (np.sqrt(box["T"][1]) - np.sqrt(box["T"][0]))
    z[:, 6] = sT ** 2
    z[:, 5] = z[:, 5] * sT  # x -> k
    return z


def sample(n, seed=0, box=BOX):
    """n scrambled-Sobol points of the box (n rounded up to a power of two for balance)."""
    m = int(np.ceil(np.log2(n)))
    return from_unit(qmc.Sobol(7, scramble=True, seed=seed).random_base2(m)[:n], box)


def inside(x, box=BOX):
    """Boolean mask of the rows of x (heston.NAMES order) that lie in the box."""
    sT = np.sqrt(x[:, 6])
    cols = [x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5] / sT, x[:, 6]]
    return np.all([(c >= box[k][0] - 1e-12) & (c <= box[k][1] + 1e-12) for c, k in zip(cols, KEYS)], axis=0)


def feller(x):
    """2 kappa theta / sigma^2: below 1 the variance can touch zero (Feller condition violated)."""
    return 2 * x[:, NAMES.index("kappa")] * x[:, NAMES.index("theta")] / x[:, NAMES.index("sigma")] ** 2
