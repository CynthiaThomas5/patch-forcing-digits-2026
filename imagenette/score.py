"""Score all generated images, retain per-image predictions and create paired figures."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.models import resnet18
from PIL import Image, ImageDraw, ImageFont
from natural import CLASS_IDS, write_csv
from sample import RUNS

NAMES = (
    "Official / dual",
    "Tuned / sync",
    "Tuned / dual",
    "Time / dual",
    "Detail / dual",
)


def mmd(x, y):
    def kernel(a, b):
        return torch.exp(-2 * torch.cdist(a, b).square()).mean()

    return (kernel(x, x) + kernel(y, y) - 2 * kernel(x, y)).item()


@torch.inference_mode()
def score(args):
    model = resnet18()
    model.load_state_dict(torch.load(args.resnet, map_location="cpu"))
    model.eval().cuda()

    def extract(path):
        d = np.load(path)
        xs = torch.from_numpy(d["rgb"]).permute(0, 3, 1, 2)
        fs, ps = [], []
        hook = model.avgpool.register_forward_hook(
            lambda m, inp, out: fs.append(out.flatten(1))
        )
        for batch in xs.split(16):
            x = F.interpolate(
                batch.cuda().float() / 255,
                (224, 224),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            x = (
                x - x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            ) / x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            ps.append(model(x).argmax(1).cpu().numpy())
        hook.remove()
        return d["labels"], np.concatenate(ps), F.normalize(torch.cat(fs), dim=1)

    ry, rp, rf = extract(args.data / "val.npz")
    rows, records = [], []
    stats = {"real_top1": float((ry == rp).mean()), "runs": {}}
    for name in RUNS:
        y, p, f = extract(args.samples / (name + ".npz"))
        cls = []
        for lab in CLASS_IDS:
            take = y == lab
            cls.append(
                (
                    int(lab),
                    int(take.sum()),
                    float((p[take] == lab).mean()),
                    mmd(f[take], rf[ry == lab]),
                )
            )
            rows.append((name, *cls[-1]))
        stats["runs"][name] = {
            "n": len(y),
            "hits": int((p == y).sum()),
            "top1": float((p == y).mean()),
            "mmd": float(np.mean([c[-1] for c in cls])),
        }
        records.extend((name, i, int(a), int(b)) for i, (a, b) in enumerate(zip(y, p)))
    write_csv(args.out / "by_class.csv", ["run", "label", "n", "top1", "mmd2"], rows)
    write_csv(
        args.out / "predictions.csv", ["run", "index", "label", "prediction"], records
    )
    with (args.out / "validation.csv").open() as file:
        val = list(csv.DictReader(file))
    stats["validation"] = {
        k: {
            m: float(np.mean([float(r[m]) for r in val if r["mode"] == k]))
            for m in ("mse", "nll", "spearman", "easy_mse")
        }
        for k in ("pf", "time", "structure")
    }
    (args.out / "summary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    gallery(args)


def gallery(args):
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 21)
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", 21)
        except OSError:
            font = ImageFont.load_default()
    bg = (247, 249, 252)
    data = [np.load(args.samples / (r + ".npz")) for r in RUNS]
    indices = [
        np.flatnonzero(data[0]["labels"] == c)[j] for c in CLASS_IDS for j in (0, 1)
    ]
    names = [
        "Springer 1",
        "Springer 2",
        "Church 1",
        "Church 2",
        "Golf ball 1",
        "Golf ball 2",
    ]
    cell = 158
    im = Image.new("RGB", (165 + cell * 6, 39 + cell * 5), bg)
    draw = ImageDraw.Draw(im)
    for j, title in enumerate(names):
        draw.text((168 + j * cell, 8), title, font=font, fill="#17478D")
    for i, (pack, name) in enumerate(zip(data, NAMES)):
        draw.text((7, 39 + i * cell + 65), name, font=font, fill="#17478D")
        for j, index in enumerate(indices):
            im.paste(
                Image.fromarray(pack["rgb"][index]).resize((cell - 2, cell - 2)),
                (165 + j * cell, 39 + i * cell),
            )
    im.save(args.out / "paired.jpg", quality=93)
    trace = np.load(args.samples / "dog_trace.npy")
    mask = np.load(args.samples / "dog_first_mask.npy")
    im = Image.new("RGB", (4 * 192, 192), bg)
    for j, frame in enumerate(trace):
        im.paste(Image.fromarray(frame).resize((192, 192)), (192 * j, 0))
    im.paste(
        Image.fromarray(np.uint8(mask) * 255).resize(
            (192, 192), Image.Resampling.NEAREST
        ),
        (576, 0),
    )
    im.save(args.out / "dog-trace.png")
    # All 48 outputs of the reference dual sampler in their original order.
    a = data[2]["rgb"]
    cols = 8
    sheet = Image.new("RGB", (cols * 112, ((len(a) + cols - 1) // cols) * 112), bg)
    for i, x in enumerate(a):
        sheet.paste(
            Image.fromarray(x).resize((112, 112)), ((i % cols) * 112, (i // cols) * 112)
        )
    sheet.save(args.out / "all-pf-dual.jpg", quality=92)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for item in ("data", "samples", "resnet"):
        p.add_argument("--" + item, type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("imagenette/results"))
    score(p.parse_args())
