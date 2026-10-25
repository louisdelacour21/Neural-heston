"""Builds the training and test sets: Sobol inputs over the box, reference prices and their exact gradients.

  python -m neural_heston.data          # writes data/train.npz (2^17 points) and data/test.npz (2^14, other seed)"""
import os
import sys
import time

import numpy as np

from . import domain, heston

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def build(n, seed, box=domain.BOX):
    x = domain.sample(n, seed=seed, box=box)
    price, grad = heston.call_and_grad(x)
    return {"x": x, "price": price, "grad": grad}


def save(name, d):
    os.makedirs(DATA, exist_ok=True)
    np.savez_compressed(os.path.join(DATA, name + ".npz"), **d)


def load(name):
    with np.load(os.path.join(DATA, name + ".npz")) as f:
        return {k: f[k] for k in f.files}


def main():
    for name, n, seed in [("train", 2 ** 17, 0), ("test", 2 ** 14, 1)]:
        t = time.time()
        save(name, build(n, seed))
        print(f"{name}: {n} points in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
