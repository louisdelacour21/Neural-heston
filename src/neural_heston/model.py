"""The neural pricer and its training by differential machine learning (Huge & Savine 2020).

The network sees the inputs rescaled to [-1, 1] over the training box (strike as k / sqrt(T), maturity as sqrt(T)) and
outputs g; the price is sqrt(T) * softplus(g), which keeps it positive and of the size of the time value at every
maturity. Activations are softplus so that the network is smooth and its input gradients (the Greeks) are meaningful.
Training fits the prices and, through a second backward pass, their gradients with respect to the seven inputs: the
"twin" loss, which acts as a strong regulariser and makes the Greeks part of what is learned rather than a by-product."""
import time

import numpy as np
import torch
from torch import nn

from .domain import BOX, KEYS


LOGS = (0, 1, 4)  # kappa, theta, v0 enter as logs: they span one to two orders of magnitude


def black(k, T, w):
    """Forward Black call with total variance w: the control variate."""
    s = torch.sqrt(w)
    d1 = (-k + w / 2) / s
    return torch.special.ndtr(d1) - torch.exp(k) * torch.special.ndtr(d1 - s)


class NeuralHeston(nn.Module):
    def __init__(self, width=128, depth=4, control=True):
        super().__init__()
        layers, n = [], 7
        for _ in range(depth):
            layers += [nn.Linear(n, width), nn.Softplus()]
            n = width
        self.net = nn.Sequential(*layers, nn.Linear(n, 1))
        lo = [BOX[k][0] for k in KEYS]
        hi = [BOX[k][1] for k in KEYS]
        lo[6], hi[6] = np.sqrt(BOX["T"][0]), np.sqrt(BOX["T"][1])  # last feature is sqrt(T)
        if control:
            for i in LOGS:
                lo[i], hi[i] = np.log(lo[i]), np.log(hi[i])
        self.register_buffer("lo", torch.tensor(lo, dtype=torch.float32))
        self.register_buffer("hi", torch.tensor(hi, dtype=torch.float32))
        self.control = control
        self.config = {"width": width, "depth": depth, "control": control}

    def forward(self, x):
        """Forward call prices for an (n, 7) tensor of inputs in heston.NAMES order.

        control=True: price = Black(k, T, w) + sqrt(T) * net, with w the Heston expected integrated variance, so the
        network only learns the smile around a Black price that already has the right level and term structure.
        control=False (first version): price = sqrt(T) * softplus(net)."""
        sT = torch.sqrt(x[:, 6:7])
        p = x[:, :5]
        if self.control:
            p = torch.cat([torch.log(p[:, :2]), p[:, 2:4], torch.log(p[:, 4:5])], dim=1)
        f = torch.cat([p, x[:, 5:6] / sT, sT], dim=1)
        f = 2 * (f - self.lo) / (self.hi - self.lo) - 1
        out = self.net(f)[:, 0]
        if not self.control:
            return sT[:, 0] * nn.functional.softplus(out)
        kappa, theta, v0, k, T = x[:, 0], x[:, 1], x[:, 4], x[:, 5], x[:, 6]
        w = theta * T + (v0 - theta) * (1 - torch.exp(-kappa * T)) / kappa
        return black(k, T, w) + 0.1 * sT[:, 0] * out

    def price_and_grad(self, x, create_graph=False):
        x = x.clone().requires_grad_(True)
        p = self(x)
        (g,) = torch.autograd.grad(p.sum(), x, create_graph=create_graph)
        return p, g

    def predict(self, x):
        """Prices and Greeks (gradients with respect to the seven inputs) as numpy arrays."""
        p, g = self.price_and_grad(torch.as_tensor(x, dtype=torch.float32))
        return p.detach().numpy().astype(float), g.detach().numpy().astype(float)


def train(data, lam=1.0, width=128, depth=4, epochs=80, batch=1024, lr=2e-3, seed=0, control=True, log=print):
    """Fits the network to {"x", "price", "grad"}. lam = 0 gives plain price regression (the baseline)."""
    torch.manual_seed(seed)
    model = NeuralHeston(width, depth, control)
    x = torch.tensor(data["x"], dtype=torch.float32)
    y = torch.tensor(data["price"], dtype=torch.float32)
    dy = torch.tensor(data["grad"], dtype=torch.float32)
    sy = y.std()
    sdy = dy.std(0)  # one scale per input so that every Greek counts, as in Huge & Savine
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps = epochs * int(np.ceil(len(x) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(x))
        tot = 0.0
        for i in range(0, len(x), batch):
            j = perm[i:i + batch]
            if lam > 0:
                p, g = model.price_and_grad(x[j], create_graph=True)
                loss = ((p - y[j]) / sy).pow(2).mean() + lam * ((g - dy[j]) / sdy).pow(2).mean()
            else:
                loss = ((model(x[j]) - y[j]) / sy).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(j)
        if log and (ep % 10 == 9 or ep == epochs - 1):
            log(f"epoch {ep + 1}/{epochs}  loss {tot / len(x):.3e}  {time.time() - t0:.0f} s")
    model.train_seconds = time.time() - t0
    return model


def save(model, path, **meta):
    torch.save({"state": model.state_dict(), "config": model.config, "meta": meta}, path)


def load(path):
    ck = torch.load(path, weights_only=False)
    model = NeuralHeston(**{"control": False, **ck["config"]})  # checkpoints from before the control variate
    model.load_state_dict(ck["state"])
    model.eval()
    model.meta = ck["meta"]
    return model
