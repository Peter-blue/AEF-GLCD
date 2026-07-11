from __future__ import print_function

import argparse
import os
import shutil
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class SceneSpec:
    source_dir: str
    image1: str
    image2: str
    label: str
    output_name: str


SCENES = {
    "California": SceneSpec(
        source_dir=os.path.join("dataset", "California"),
        image1="California_t1.bmp",
        image2="California_t2.bmp",
        label="California_gt.bmp",
        output_name="CaliforniaPatch",
    ),
    "Shuguang": SceneSpec(
        source_dir=os.path.join("dataset", "Shuguang"),
        image1="shuguang_1.bmp",
        image2="shuguang_2.bmp",
        label="shuguang_gt.bmp",
        output_name="ShuguangPatch",
    ),
}


def load_rgb(path):
    return Image.open(path).convert("RGB")


def load_label(path):
    return Image.open(path).convert("L")


def positions(length, patch_size, stride):
    if length < patch_size:
        raise ValueError(f"Image side {length} is smaller than patch size {patch_size}.")
    values = list(range(0, max(1, length - patch_size + 1), stride))
    last = length - patch_size
    if values[-1] != last:
        values.append(last)
    return sorted(set(values))


def choose_split_axis(width, height, patch_size, xs, ys, split_ratio):
    candidates = []
    for axis, length, starts, other_count in (
        ("x", width, xs, len(ys)),
        ("y", height, ys, len(xs)),
    ):
        split = int(round(length * split_ratio))
        train = [s for s in starts if s + patch_size <= split]
        test = [s for s in starts if s >= split]
        candidates.append((len(train) * other_count, len(test) * other_count, axis, split))
    valid = [c for c in candidates if c[0] > 0 and c[1] > 0]
    if not valid:
        raise ValueError("Cannot build a non-overlapping train/test split with current patch parameters.")
    # Prefer a split with more test samples while keeping enough training samples.
    return max(valid, key=lambda item: (min(item[0], item[1]), item[0] + item[1]))


def assign_split(x, y, patch_size, axis, split):
    start = x if axis == "x" else y
    if start + patch_size <= split:
        return "train"
    if start >= split:
        return "test"
    return None


def recreate_dirs(root):
    if os.path.exists(root):
        shutil.rmtree(root)
    for split in ("train", "test"):
        for sub in ("Image", "Image2", "label"):
            os.makedirs(os.path.join(root, split, sub), exist_ok=True)


def binarize_label(label_patch):
    arr = np.array(label_patch)
    arr = np.where(arr > 0, 255, 0).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def prepare_scene(name, patch_size, stride, split_ratio, output_name=None):
    if name not in SCENES:
        raise ValueError(f"Unknown scene {name}. Available: {', '.join(sorted(SCENES))}")
    spec = SCENES[name]
    out_name = output_name or spec.output_name
    out_root = os.path.join("dataset", out_name)

    img1 = load_rgb(os.path.join(spec.source_dir, spec.image1))
    img2 = load_rgb(os.path.join(spec.source_dir, spec.image2))
    label = load_label(os.path.join(spec.source_dir, spec.label))
    if img1.size != img2.size or img1.size != label.size:
        raise ValueError(f"Image size mismatch in {name}: {img1.size}, {img2.size}, {label.size}")

    width, height = img1.size
    xs = positions(width, patch_size, stride)
    ys = positions(height, patch_size, stride)
    train_count, test_count, axis, split = choose_split_axis(width, height, patch_size, xs, ys, split_ratio)

    recreate_dirs(out_root)
    counts = {"train": 0, "test": 0, "discarded_boundary": 0}
    for y in ys:
        for x in xs:
            split_name = assign_split(x, y, patch_size, axis, split)
            if split_name is None:
                counts["discarded_boundary"] += 1
                continue
            box = (x, y, x + patch_size, y + patch_size)
            stem = f"{name.lower()}_x{x:04d}_y{y:04d}.png"
            img1.crop(box).save(os.path.join(out_root, split_name, "Image", stem))
            img2.crop(box).save(os.path.join(out_root, split_name, "Image2", stem))
            binarize_label(label.crop(box)).save(os.path.join(out_root, split_name, "label", stem))
            counts[split_name] += 1

    print(f"Prepared {name} -> {out_root}")
    print(f"  image_size={width}x{height}, patch_size={patch_size}, stride={stride}")
    print(f"  split_axis={axis}, split_position={split}, split_ratio={split_ratio}")
    print(f"  train={counts['train']}, test={counts['test']}, discarded_boundary={counts['discarded_boundary']}")
    return out_root, counts


def main():
    parser = argparse.ArgumentParser(description="Prepare patch datasets from single-scene CD images.")
    parser.add_argument("--scene", choices=sorted(SCENES), required=True)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--split_ratio", type=float, default=0.70)
    parser.add_argument("--output_name", type=str, default=None)
    args = parser.parse_args()
    prepare_scene(args.scene, args.patch_size, args.stride, args.split_ratio, args.output_name)


if __name__ == "__main__":
    main()
