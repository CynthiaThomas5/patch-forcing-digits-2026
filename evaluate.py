"""Read checkpoints, generate matched images, evaluate and plot all runs."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from experiment import MODES, data, pixels, sample

SEEDS = (11, 22, 33)
NAMES = {"pf": "Patch Forcing", "global": "Global prior", "class": "Class prior"}


def conditional_metrics(gen, labels, real, y):
    mmd, spread = [], []
    for k in range(10):
        a, b = gen[labels == k], real[y == k]
        def kernel(u, v):
            return rbf_kernel(u, v, gamma=1/32).mean()
        mmd.append(kernel(a, a) + kernel(b, b) - 2 * kernel(a, b))
        spread.append(a.var(0).mean())
    return np.mean(mmd), np.mean(spread)


def evaluate(runs, out):
    out.mkdir(parents=True, exist_ok=True)
    train, test, y, ytest = data()
    train, test = pixels(train), pixels(test)
    clf = SVC(C=10).fit(train, y)
    labels = np.tile(np.arange(10), 100)
    noise = np.random.default_rng(1201).standard_normal((len(labels), 16, 4), dtype=np.float32)
    rows, examples = [], []
    for seed in SEEDS:
        for mode in MODES:
            with np.load(runs / f"{mode}-{seed}/checkpoint.npz") as w:
                for policy in ("dual", "parallel"):
                    gen = sample(dict(w), labels, noise, policy)
                    mmd, spread = conditional_metrics(gen, labels, test, ytest)
                    hit = np.mean(clf.predict(gen) == labels)
                    rows.append((seed, mode, policy, hit, mmd, spread))
                    if seed == SEEDS[0] and policy == "dual":
                        examples.append(gen[:20].copy())
    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "mode", "sampler", "hit", "mmd2", "within_var"])
        writer.writerows(rows)
    real_var = np.mean([test[ytest == k].var(0).mean() for k in range(10)])
    with np.load(runs / "class-11/checkpoint.npz") as w:
        # Direct endpoint draws from the diagonal Gaussian prior, before neural correction.
        draws = pixels(w["mu"][labels] + np.sqrt(w["variance"][labels]) * noise).clip(-1, 1)
        mmd_diag, var_diag = conditional_metrics(draws, labels, test, ytest)
        prior_hit = np.mean(clf.predict(draws) == labels)
    (out / "prior_only.txt").write_text(
        f"Class prior alone: hit={prior_hit:.6f}, MMD2={mmd_diag:.6f}, within_var={var_diag:.6f}\n")
    (out / "reference.txt").write_text(f"Real test SVM hit: {clf.score(test,ytest):.6f}\n"
                                       f"Real test class-conditional variance: {real_var:.6f}\n")
    # Two consecutive draws per label. No selection using generated quality.
    fig, axes = plt.subplots(6, 10, figsize=(9.5, 5.1))
    for m, (mode, gen) in enumerate(zip(MODES, examples)):
        for r in range(2):
            for k in range(10):
                ax = axes[2*m+r, k]
                ax.imshow(gen[10*r+k].reshape(8, 8), vmin=-1, vmax=1,
                          interpolation="nearest", cmap="gray")
                ax.set(xticks=[], yticks=[])
                if m == 0 and r == 0:
                    ax.set_title(str(k))
                if k == 0 and r == 0:
                    ax.set_ylabel(NAMES[mode], rotation=0, ha="right", va="center")
    fig.subplots_adjust(left=.16, bottom=.02, right=.99, top=.93, hspace=.12, wspace=.1)
    fig.savefig(out / "paired.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for mode in MODES:
        traces = [np.genfromtxt(runs / f"{mode}-{s}/loss.csv", delimiter=",", names=True)
                  for s in SEEDS]
        ax.plot(traces[0]["step"], np.mean([a["velocity_mse_last_batch"] for a in traces],0),
                label=NAMES[mode])
    ax.set(xlabel="Training step", ylabel="Velocity MSE", yscale="log")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "loss.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()
    with threadpool_limits(limits=1):
        evaluate(args.runs, args.out)

