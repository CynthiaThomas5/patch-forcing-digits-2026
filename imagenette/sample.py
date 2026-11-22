"""Generate paired class-conditional images at a matched evaluation budget."""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from natural import (
    CLASS_IDS,
    Readout,
    backbone,
    features,
    image_times,
    load_heads,
    read_vae,
)

RUNS = ("official_dual", "pf_sync", "pf_dual", "time_dual", "structure_dual")


@torch.inference_mode()
def generate(args):
    model = backbone(args)
    heads = load_heads(model, args.out)
    heads["official"] = Readout(model.final_layer.linear, "pf").cuda().eval()
    vae = read_vae(args.vae)
    args.samples.mkdir(parents=True, exist_ok=True)
    ys = torch.tensor(CLASS_IDS).repeat_interleave(args.per_class)
    noises = torch.randn(
        len(ys), 4, 32, 32, generator=torch.Generator().manual_seed(args.seed)
    )

    def decode(z):
        pixels = vae.decode(z.half() / 0.18215).sample.float()
        return (
            ((pixels.clamp(-1, 1) + 1) * 127.5)
            .round()
            .byte()
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )

    for name in RUNS:
        headname, schedule = name.split("_")
        head = heads[headname]
        output, trace = [], []
        for i in range(0, len(ys), args.batch):
            x = noises[i : i + args.batch].cuda()
            y = ys[i : i + args.batch].cuda()
            conditions = torch.cat([torch.full_like(y, 1000), y])

            def forward(t):
                xx, tt = torch.cat([x, x]), torch.cat([t, t])
                h = features(model, xx, tt, conditions)
                v, u = head(model, h, xx, tt)
                vu, vc = v.chunk(2)
                uu, uc = u.chunk(2)
                return vu + args.cfg * (vc - vu), uu + args.cfg * (uc - uu)

            steps = 24 if schedule == "sync" else 6
            for outer in range(steps):
                t = torch.full((len(x), 256), outer / steps, device=x.device)
                v, u = forward(t)
                if schedule == "sync":
                    x = x + v / steps
                else:
                    score = F.avg_pool2d(u.exp(), 2).flatten(1)
                    mask = score < torch.quantile(score, 0.4, dim=1, keepdim=True)
                    dt = torch.where(mask, 1 / steps, 1 / (steps * 4))
                    x = x + image_times(dt) * v
                    t = t + dt
                    inner = (~mask).float() / (steps * 4)
                    for _ in range(3):
                        v, _ = forward(t)
                        x = x + image_times(inner) * v
                        t = t + inner
                if name == "pf_dual" and i == 0 and outer in (0, 2, 5):
                    trace.append(decode(x[:1])[0])
                    if outer == 0:
                        np.save(
                            args.samples / "dog_first_mask.npy",
                            mask[0].cpu().numpy().reshape(16, 16),
                        )
            output.append(decode(x))
        np.savez(
            args.samples / (name + ".npz"),
            rgb=np.concatenate(output),
            labels=ys.numpy(),
        )
        if trace:
            np.save(args.samples / "dog_trace.npy", np.stack(trace))
        print(name, len(ys), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for opt in ("upstream", "checkpoint", "vae", "samples"):
        p.add_argument("--" + opt, type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("imagenette/results"))
    p.add_argument("--per-class", type=int, default=16)
    p.add_argument("--seed", type=int, default=1201)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--cfg", type=float, default=4)
    generate(p.parse_args())
