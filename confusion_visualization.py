from __future__ import print_function

import os

import numpy as np
from PIL import Image, ImageDraw


CONFUSION_COLORS = {
    "TN": (0, 0, 0),
    "TP": (255, 255, 255),
    "FP": (255, 0, 0),
    "FN": (0, 200, 0),
    "IGNORE": (128, 128, 128),
}


def to_binary(array):
    array = np.asarray(array)
    if array.ndim == 3:
        array = array[..., 0]
    return (array > 0).astype(np.uint8)


def build_confusion_map(prediction, label, valid_mask=None):
    """Create an RGB map with TN black, TP white, FP red and FN green."""
    pred = to_binary(prediction)
    target = to_binary(label)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/label shape mismatch: {pred.shape} vs {target.shape}")

    rgb = np.zeros(pred.shape + (3,), dtype=np.uint8)
    rgb[np.logical_and(pred == 0, target == 0)] = CONFUSION_COLORS["TN"]
    rgb[np.logical_and(pred == 1, target == 1)] = CONFUSION_COLORS["TP"]
    rgb[np.logical_and(pred == 1, target == 0)] = CONFUSION_COLORS["FP"]
    rgb[np.logical_and(pred == 0, target == 1)] = CONFUSION_COLORS["FN"]
    if valid_mask is not None:
        valid = to_binary(valid_mask).astype(bool)
        if valid.shape != pred.shape:
            raise ValueError(f"Valid-mask shape mismatch: {valid.shape} vs {pred.shape}")
        rgb[~valid] = CONFUSION_COLORS["IGNORE"]
    return rgb


def confusion_counts(prediction, label, valid_mask=None):
    pred = to_binary(prediction)
    target = to_binary(label)
    if valid_mask is not None:
        valid = to_binary(valid_mask).astype(bool)
        pred = pred[valid]
        target = target[valid]
    return {
        "TN": int(np.logical_and(pred == 0, target == 0).sum()),
        "TP": int(np.logical_and(pred == 1, target == 1).sum()),
        "FP": int(np.logical_and(pred == 1, target == 0).sum()),
        "FN": int(np.logical_and(pred == 0, target == 1).sum()),
    }


def save_confusion_map(prediction, label, path, valid_mask=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rgb = build_confusion_map(prediction, label, valid_mask=valid_mask)
    Image.fromarray(rgb).save(path)
    return confusion_counts(prediction, label, valid_mask=valid_mask)


def make_legend(height=32, item_width=112, include_ignore=False):
    keys = ["TP", "TN", "FP", "FN"]
    if include_ignore:
        keys.append("IGNORE")
    canvas = Image.new("RGB", (item_width * len(keys), height), "white")
    draw = ImageDraw.Draw(canvas)
    box = max(12, height - 12)
    for index, key in enumerate(keys):
        x = index * item_width + 6
        y = (height - box) // 2
        draw.rectangle((x, y, x + box, y + box), fill=CONFUSION_COLORS[key], outline=(80, 80, 80))
        draw.text((x + box + 7, max(2, y + 1)), key, fill=(20, 20, 20))
    return canvas


def resize_contain(image, target_width, target_height, background=(255, 255, 255)):
    image = image.convert("RGB")
    scale = min(target_width / float(image.width), target_height / float(image.height))
    resized = image.resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        Image.Resampling.NEAREST,
    )
    canvas = Image.new("RGB", (target_width, target_height), background)
    x = (target_width - resized.width) // 2
    y = (target_height - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def save_labeled_panel(items, path, tile_size=(320, 320), title=None, include_legend=True):
    if not items:
        raise ValueError("Panel requires at least one item.")
    tile_w, tile_h = tile_size
    header_h = 34
    title_h = 38 if title else 0
    legend = make_legend() if include_legend else None
    legend_h = legend.height + 10 if legend is not None else 0
    width = tile_w * len(items)
    height = title_h + header_h + tile_h + legend_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    if title:
        draw.text((10, 10), str(title), fill=(20, 20, 20))
    top = title_h
    for index, (name, image) in enumerate(items):
        x = index * tile_w
        draw.rectangle((x, top, x + tile_w, top + header_h), fill=(245, 245, 245))
        draw.text((x + 8, top + 9), str(name), fill=(20, 20, 20))
        tile = resize_contain(image, tile_w, tile_h, background=(255, 255, 255))
        canvas.paste(tile, (x, top + header_h))

    if legend is not None:
        x = max(0, (width - legend.width) // 2)
        canvas.paste(legend, (x, height - legend_h + 5))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)

