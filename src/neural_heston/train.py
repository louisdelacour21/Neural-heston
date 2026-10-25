"""Trains the pricer on data/train.npz and reports its test error.

  python -m neural_heston.train                    # differential ML, writes models/dml.pt
  python -m neural_heston.train --lam 0 --name plain   # price-only baseline
  python -m neural_heston.train --n 4096 --name dml_4k   # smaller training set"""
import argparse
import os
import sys

import numpy as np
import torch

from . import data, model

MODELS = os.path.join(os.path.dirname(__file__), "..", "..", "models")


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--lam", type=float, default=1.0)
    a.add_argument("--n", type=int, default=0, help="first n training points (0 = all)")
    a.add_argument("--epochs", type=int, default=80)
    a.add_argument("--width", type=int, default=128)
    a.add_argument("--depth", type=int, default=4)
    a.add_argument("--name", default="dml")
    args = a.parse_args(argv)
    torch.set_num_threads(8)  # measured fastest on this 22-core CPU: more threads only add overhead for a small MLP
    tr = data.load("train")
    full = len(tr["x"])
    if args.n:
        tr = {k: v[:args.n] for k, v in tr.items()}
    epochs = args.epochs * full // len(tr["x"])  # same number of optimiser steps whatever the training-set size
    m = model.train(tr, lam=args.lam, width=args.width, depth=args.depth, epochs=epochs)
    te = data.load("test")
    p, _ = m.predict(te["x"])
    err = p - te["price"]
    print(f"test RMSE {np.sqrt(np.mean(err ** 2)):.2e}  max |err| {np.abs(err).max():.2e}  ({m.train_seconds:.0f} s)")
    os.makedirs(MODELS, exist_ok=True)
    model.save(m, os.path.join(MODELS, args.name + ".pt"), lam=args.lam, n=len(tr["x"]), epochs=epochs,
               train_seconds=m.train_seconds)


if __name__ == "__main__":
    sys.exit(main())
