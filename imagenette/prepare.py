"""Prepare a seeded, clean-label subset of the official Imagenette v2 splits."""

import argparse
import csv
import hashlib
import io
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

CLASSES = (
    ("n02102040", 217, "English springer"),
    ("n03028079", 497, "church"),
    ("n03445777", 574, "golf ball"),
)
ARCHIVE_SHA256 = "64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5"


def prepare(archive, out):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Use the official imagenette2-160.tgz release")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(23)
    manifest = []
    with tarfile.open(archive) as tar:
        files = [
            m
            for m in tar.getmembers()
            if m.isfile() and m.name.lower().endswith(".jpeg")
        ]
        for split, count in [("train", 96), ("val", 32)]:
            images, labels = [], []
            for synset, label, _ in CLASSES:
                candidates = sorted(
                    (
                        m
                        for m in files
                        if m.name.split("/")[-3:-1] == [split, synset]
                        and not Path(m.name).name.startswith(".")
                    ),
                    key=lambda m: m.name,
                )
                for idx in rng.choice(len(candidates), count, replace=False):
                    member = candidates[idx]
                    im = Image.open(io.BytesIO(tar.extractfile(member).read())).convert(
                        "RGB"
                    )
                    images.append(
                        np.asarray(
                            ImageOps.fit(
                                im, (256, 256), method=Image.Resampling.BICUBIC
                            )
                        )
                    )
                    labels.append(label)
                    manifest.append((split, synset, label, member.name))
            np.savez(
                out / f"{split}.npz", rgb=np.stack(images), labels=np.array(labels)
            )
    with (out / "manifest.csv").open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(("split", "synset", "imagenet_id", "archive_path"))
        writer.writerows(manifest)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    prepare(args.archive, args.out)
