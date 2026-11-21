"""Train and sample a small patch-time conditional flow on sklearn Digits."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import expit, ndtr, ndtri
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

MODES = ("pf", "global", "class")
LABELS = np.eye(10, dtype=np.float32)


def data():
    digits = load_digits()
    tiles = (digits.images.astype("f") / 8 - 1).reshape(-1, 4, 2, 4, 2)
    tiles = tiles.transpose(0, 1, 3, 2, 4).reshape(-1, 16, 4)
    return train_test_split(tiles, digits.target, test_size=.2, random_state=23,
                            stratify=digits.target)


def pixels(tiles):
    return tiles.reshape(-1, 4, 4, 2, 2).transpose(0, 1, 3, 2, 4).reshape(-1, 64)


def patch_times(rng, batch):
    upper = expit(.5 + rng.standard_normal((batch, 1)))
    lower_cdf = ndtr(-upper / .2)
    quantiles = rng.uniform(size=(batch, 16)) * (.5 - lower_cdf) + lower_cdf
    return (upper + .2 * ndtri(quantiles)).astype("f")


def moments(x, y, mode):
    # Standard-normal reference gives a shared analytic skip for the PF baseline.
    if mode == "pf":
        return np.zeros((10, 16, 4), "f"), np.ones((10, 16, 4), "f")
    groups = [x[y == k] if mode == "class" else x for k in range(10)]
    return (np.stack([g.mean(0) for g in groups]),
            np.stack([np.maximum(g.var(0), .02) for g in groups]))


def forward(weights, mu, variance, state, times, labels):
    inp = np.concatenate((state.reshape(-1, 64), times, LABELS[labels]), axis=1)
    hidden = np.tanh(inp @ weights["W"] + weights["b"])
    pred = hidden @ weights["V"] + weights["c"]
    t = times[:, :, None]
    m, var = mu[labels], variance[labels]
    gain = (t * var - (1 - t)) / ((1 - t)**2 + t**2 * var)
    skip = m + gain * (state - t * m)
    velocity = skip + pred[:, :64].reshape(-1, 16, 4)
    return velocity, pred[:, 64:], hidden, inp


def train(mode, seed, steps, out):
    x, _, y, _ = data()
    mu, var = moments(x, y, mode)
    rng = np.random.default_rng(seed)
    weights = dict(W=(rng.standard_normal((90, 128)) / np.sqrt(90)).astype("f"),
                   b=np.zeros(128, "f"),
                   V=(rng.standard_normal((128, 80)) * .01).astype("f"),
                   c=np.zeros(80, "f"))
    first = {k: np.zeros_like(v) for k, v in weights.items()}
    second = {k: np.zeros_like(v) for k, v in weights.items()}
    folder = out / f"{mode}-{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with (folder / "loss.csv").open("w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["step", "velocity_mse_last_batch", "difficulty_nll_last_batch"])
        for step in range(1, steps + 1):
            ids = rng.integers(len(x), size=96)
            clean, labels = x[ids], y[ids]
            t = patch_times(rng, len(ids))
            noise = rng.standard_normal(clean.shape, dtype=np.float32)
            target = clean - noise
            state = noise + t[:, :, None] * target
            v, raw, h, inp = forward(weights, mu, var, state, t, labels)
            err = v - target
            logvar = np.clip(raw, -7, 4)
            mse = (err**2).mean(axis=2)
            ratio = mse * np.exp(-logvar)
            # NLL uses detached velocity residuals. Only variance outputs receive its gradient.
            dv = 2 * err.reshape(-1, 64) / err.size
            ds = .005 * (1 - ratio) * ((raw > -7) & (raw < 4)) / raw.size
            dout = np.concatenate((dv, ds), axis=1)
            dh = (dout @ weights["V"].T) * (1 - h*h)
            grad = dict(W=inp.T @ dh, b=dh.sum(0), V=h.T @ dout, c=dout.sum(0))
            for k, g in grad.items():
                first[k] = .9 * first[k] + .1 * g
                second[k] = .999 * second[k] + .001 * g*g
                weights[k] -= .001 * (first[k] / (1 - .9**step)) / (
                    np.sqrt(second[k] / (1 - .999**step)) + 1e-8)
            if step == 1 or step % 100 == 0 or step == steps:
                log.writerow([step, float(mse.mean()), float(.5*(ratio + logvar).mean())])
    elapsed = time.perf_counter() - start
    np.savez_compressed(folder / "checkpoint.npz", **weights, mu=mu, variance=var)
    (folder / "config.json").write_text(json.dumps(dict(mode=mode, seed=seed, steps=steps,
        batch=96, hidden=128, lr=.001, split_seed=23, ltg_m=.5, ltg_s=1., ltg_sigma=.2,
        nll_weight=.01, variance_floor=.02, seconds=elapsed), indent=2))
    print(folder, round(elapsed, 2), flush=True)


def sample(weights, labels, noise, policy="dual", rounds=8):
    x = noise.copy()
    t = np.zeros(x.shape[:2], "f")
    dt = 1 / rounds
    for _ in range(rounds):
        v, score, _, _ = forward(weights, weights["mu"], weights["variance"], x, t, labels)
        delta = np.full_like(t, dt / 2)
        if policy == "dual":
            easy = np.argsort(score, axis=1)[:, :8]
            np.put_along_axis(delta, easy, dt, axis=1)
        x += delta[:, :, None] * v
        t += delta
        v, _, _, _ = forward(weights, weights["mu"], weights["variance"], x, t, labels)
        remaining = dt - delta
        x += remaining[:, :, None] * v
        t += remaining
    return pixels(x).clip(-1, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        train(args.mode, args.seed, args.steps, args.out)

