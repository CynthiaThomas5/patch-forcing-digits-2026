"""Fine-tune PFT-B output projections and evaluate patch uncertainty on Imagenette."""

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MODES = ("pf", "time", "structure")
CLASS_IDS = (217, 497, 574)


def read_vae(path):
    from diffusers import AutoencoderKL

    return (
        AutoencoderKL.from_pretrained(str(path), local_files_only=True)
        .eval()
        .to("cuda", torch.float16)
    )


def encode(args):
    vae = read_vae(args.vae)
    for split in ("train", "val"):
        with np.load(args.data / f"{split}.npz") as data:
            rgb = torch.from_numpy(data["rgb"]).permute(0, 3, 1, 2)
            codes = []
            with torch.inference_mode():
                for batch in rgb.split(4):
                    images = batch.to("cuda", torch.float16) / 127.5 - 1
                    # Use posterior means for a reproducible small-data comparison.
                    codes.append(
                        (vae.encode(images).latent_dist.mean * 0.18215).float().cpu()
                    )
            torch.save(
                (torch.cat(codes), torch.from_numpy(data["labels"])),
                args.data / f"{split}.pt",
            )


def backbone(args):
    sys.path.insert(0, str(args.upstream.resolve()))
    from patch_flow.models.pf_transformer import PF_B_2

    m = PF_B_2(compile=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    m.load_state_dict(ckpt["state_dict"], strict=True)
    return m.eval().requires_grad_(False).cuda().half()


def image_times(t):
    return t.reshape(-1, 1, 16, 16).repeat_interleave(2, 2).repeat_interleave(2, 3)


def training_tuple(clean):
    cap = torch.sigmoid(0.7 + torch.randn(len(clean), 1, device=clean.device))
    t = cap - torch.randn(len(clean), 256, device=clean.device).abs() * (cap / 2).clamp(
        max=0.6
    )
    t = torch.where(t < 0, torch.rand_like(t) * cap, t)
    noise = torch.randn_like(clean)
    target = clean - noise
    return noise + image_times(t) * target, t, target


class Readout(nn.Module):
    def __init__(self, projection, mode):
        super().__init__()
        from copy import deepcopy

        self.linear = deepcopy(projection).float().requires_grad_(True)
        self.mode = mode
        if mode != "pf":
            self.calibrate = nn.Conv2d(2 if mode == "time" else 3, 1, 1)
            nn.init.zeros_(self.calibrate.weight)
            nn.init.zeros_(self.calibrate.bias)

    def forward(self, model, features, x, t):
        raw = model.unpatchify(self.linear(features.float()))
        v, logvar = raw[:, :4], raw[:, 4:]
        if self.mode != "pf":
            ti = image_times(t)
            cues = [logvar.detach(), ti]
            if self.mode == "structure":
                # Estimate the endpoint, then local high-frequency energy in latent space.
                endpoint = x.float() + (1 - ti) * v.detach()
                padded = F.pad(endpoint, (1, 1, 1, 1), mode="replicate")
                detail = (
                    (endpoint - F.avg_pool2d(padded, 3, stride=1))
                    .square()
                    .mean(1, keepdim=True)
                )
                cues.append(torch.log1p(detail))
            logvar = logvar + self.calibrate(torch.cat(cues, 1))
        return v, logvar


def features(model, x, t, y):
    box = []
    hook = model.final_layer.linear.register_forward_pre_hook(
        lambda mod, inp: box.append(inp[0].detach())
    )
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            model(x, t, y)
    finally:
        hook.remove()
    return box[0]


def losses(v, u, target):
    sq = (v - target).square().mean(1, keepdim=True)
    logvar = u.clamp(-10, 8)
    return sq.mean(), (0.5 * (logvar + sq.detach() * (-logvar).exp())).mean()


def train(args):
    torch.manual_seed(args.seed)
    model = backbone(args)
    x, y = torch.load(args.data / "train.pt")
    x, y = x.cuda(), y.cuda()
    heads = {mode: Readout(model.final_layer.linear, mode).cuda() for mode in MODES}
    opt = {
        mode: torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0)
        for mode, head in heads.items()
    }
    args.out.mkdir(parents=True, exist_ok=True)
    steps = []
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        ids = torch.randint(len(x), (args.batch,), device="cuda")
        noisy, t, target = training_tuple(x[ids])
        h = features(model, noisy, t, y[ids])
        for mode, head in heads.items():
            v, u = head(model, h, noisy, t)
            mse, nll = losses(v, u, target)
            opt[mode].zero_grad()
            (mse + 0.01 * nll).backward()
            opt[mode].step()
            if step == 1 or step % 10 == 0 or step == args.steps:
                steps.append((step, mode, mse.item(), nll.item()))
        if step % 50 == 0:
            print(f"step {step}/{args.steps}", flush=True)
    torch.cuda.synchronize()
    base = model.final_layer.linear.state_dict()
    for mode, head in heads.items():
        delta = {
            key: (
                value - base[key[7:]].float() if key.startswith("linear.") else value
            ).cpu()
            for key, value in head.state_dict().items()
        }
        torch.save(delta, args.out / f"{mode}.pt")
    write_csv(
        args.out / "loss.csv", ["step", "mode", "velocity_mse", "gaussian_nll"], steps
    )
    config = dict(
        steps=args.steps,
        batch=args.batch,
        seed=args.seed,
        lr=args.lr,
        nll_weight=0.01,
        seconds=time.perf_counter() - start,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        python=platform.python_version(),
        parameters={
            k: sum(p.numel() for p in v.parameters()) for k, v in heads.items()
        },
    )
    (args.out / "config.json").write_text(json.dumps(config, indent=2))


def write_csv(file, header, rows):
    with file.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        writer.writerows(rows)


def load_heads(model, directory):
    heads = {}
    for mode in MODES:
        head = Readout(model.final_layer.linear, mode).cuda()
        delta = torch.load(directory / f"{mode}.pt", map_location="cpu")
        initial = head.state_dict()
        head.load_state_dict(
            {
                k: initial[k] + v.cuda() if k.startswith("linear.") else v
                for k, v in delta.items()
            }
        )
        heads[mode] = head.eval()
    return heads


@torch.inference_mode()
def validate(args):
    from scipy.stats import spearmanr

    torch.manual_seed(9001)
    model = backbone(args)
    heads = load_heads(model, args.out)
    data, labels = torch.load(args.data / "val.pt")
    rows = []
    for begin in range(0, len(data), args.batch):
        x = data[begin : begin + args.batch].cuda()
        y = labels[begin : begin + args.batch].cuda()
        noisy, t, target = training_tuple(x)
        h = features(model, noisy, t, y)
        for mode, head in heads.items():
            v, u = head(model, h, noisy, t)
            err = (v - target).square().mean(1, keepdim=True)
            u = u.clamp(-10, 8)
            nll = 0.5 * (u + err * (-u).exp())
            score = F.avg_pool2d(u.exp(), 2).flatten(1).cpu().numpy()
            pe = F.avg_pool2d(err, 2).flatten(1).cpu().numpy()
            for j in range(len(x)):
                order = np.argsort(score[j])[:102]
                rows.append(
                    (
                        begin + j,
                        int(y[j]),
                        mode,
                        float(pe[j].mean()),
                        nll[j].mean().item(),
                        spearmanr(pe[j], score[j]).statistic,
                        float(pe[j, order].mean()),
                    )
                )
    write_csv(
        args.out / "validation.csv",
        ["index", "label", "mode", "mse", "nll", "spearman", "easy_mse"],
        rows,
    )


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("encode", "train", "validate"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--upstream", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--vae", type=Path)
    p.add_argument("--out", type=Path, default=Path("imagenette/results"))
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=23)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    {"train": train, "encode": encode, "validate": validate}[args.action](args)
