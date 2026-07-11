from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from PIL import Image

from test import save_visual_panel, str2bool
from test_full_scene import (
    SCENES,
    build_models,
    cd_to_uint8,
    load_label_np,
    load_rgb_np,
    metric_row,
    np_to_model_tensor,
    pad_to_patch,
    positions,
    resolve_optional,
    resolve_speckle_mode,
    tensor_to_image_np,
    to_uint8_img,
)
from utils import is_image_file


def parse_shift_list(value):
    """Parse '0,0;2,0;4,0;8,0' or '0;2;4;8' into (dx, dy) pairs."""
    shifts = []
    for token in str(value).replace("|", ";").split(";"):
        token = token.strip()
        if not token:
            continue
        if "," in token:
            dx, dy = token.split(",", 1)
            shifts.append((int(dx.strip()), int(dy.strip())))
        else:
            shifts.append((int(token), 0))
    if not shifts:
        raise ValueError("shift_list is empty.")
    return shifts


def shift_tag(dx, dy):
    return f"dx{dx:+d}_dy{dy:+d}".replace("+", "p").replace("-", "m")


def shift_tensor(x, dx, dy, fill_mode="edge"):
    """Shift BCHW tensor by integer pixels while preserving the original size.

    Positive dx shifts content to the right; positive dy shifts content downward.
    """
    dx = int(dx)
    dy = int(dy)
    if dx == 0 and dy == 0:
        return x
    if x.dim() != 4:
        raise ValueError("shift_tensor expects a BCHW tensor.")

    _, _, h, w = x.shape
    pad_left = max(dx, 0)
    pad_right = max(-dx, 0)
    pad_top = max(dy, 0)
    pad_bottom = max(-dy, 0)
    crop_x = max(-dx, 0)
    crop_y = max(-dy, 0)

    if fill_mode == "edge":
        padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
    elif fill_mode == "zero":
        padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
    else:
        raise ValueError(f"Unknown fill_mode: {fill_mode}")
    return padded[:, :, crop_y : crop_y + h, crop_x : crop_x + w]


def shift_numpy(x, dx, dy, fill_mode="edge"):
    """Shift HW/HWC numpy array using the same convention as shift_tensor."""
    dx = int(dx)
    dy = int(dy)
    if dx == 0 and dy == 0:
        return x.copy()
    if x.ndim not in (2, 3):
        raise ValueError("shift_numpy expects an HW or HWC array.")

    h, w = x.shape[:2]
    pad_left = max(dx, 0)
    pad_right = max(-dx, 0)
    pad_top = max(dy, 0)
    pad_bottom = max(-dy, 0)
    crop_x = max(-dx, 0)
    crop_y = max(-dy, 0)
    pad_width = [(pad_top, pad_bottom), (pad_left, pad_right)]
    if x.ndim == 3:
        pad_width.append((0, 0))
    if fill_mode == "edge":
        padded = np.pad(x, pad_width, mode="edge")
    elif fill_mode == "zero":
        padded = np.pad(x, pad_width, mode="constant", constant_values=0)
    else:
        raise ValueError(f"Unknown fill_mode: {fill_mode}")
    return padded[crop_y : crop_y + h, crop_x : crop_x + w].copy()


def maybe_shift_generated(fake_opt, fake_sar, dx, dy, branch, fill_mode):
    if branch in ("both", "optical"):
        fake_opt = shift_tensor(fake_opt, dx, dy, fill_mode)
    if branch in ("both", "sar"):
        fake_sar = shift_tensor(fake_sar, dx, dy, fill_mode)
    return fake_opt, fake_sar


def prepare_input_misalignment(image1, image2, label, dx, dy, modality, fill_mode):
    """Shift one original modality and prepare branch-specific labels.

    Optical-domain CD is evaluated in Image1 coordinates; SAR-domain CD is
    evaluated in Image2 coordinates. The valid mask removes padded borders.
    """
    valid = np.ones(label.shape[:2], dtype=np.uint8)
    shifted_valid = shift_numpy(valid, dx, dy, fill_mode="zero")
    label_opt = label.copy()
    label_sar = label.copy()
    if modality == "sar":
        image2 = shift_numpy(image2, dx, dy, fill_mode)
        label_sar = shift_numpy(label, dx, dy, fill_mode="zero")
    elif modality == "optical":
        image1 = shift_numpy(image1, dx, dy, fill_mode)
        label_opt = shift_numpy(label, dx, dy, fill_mode="zero")
    else:
        raise ValueError(f"Unknown input_shift_modality: {modality}")
    return image1, image2, label_opt, label_sar, shifted_valid


def confusion_from_np(pred, label, valid_mask=None):
    pred = pred.astype(np.uint8)
    label = label.astype(np.uint8)
    if valid_mask is not None:
        valid_mask = valid_mask.astype(bool)
        pred = pred[valid_mask]
        label = label[valid_mask]
    return {
        "tp": float(np.logical_and(pred == 1, label == 1).sum()),
        "tn": float(np.logical_and(pred == 0, label == 0).sum()),
        "fp": float(np.logical_and(pred == 1, label == 0).sum()),
        "fn": float(np.logical_and(pred == 0, label == 1).sum()),
    }


def metrics_from_counts(counts, eps=1e-8):
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    oa = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "oa": oa,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def accumulate_counts(total, current):
    for key in ("tp", "tn", "fp", "fn"):
        total[key] += current[key]


def row_from_metrics(dx, dy, opt_metrics, sar_metrics):
    mean_f1 = 0.5 * (opt_metrics["f1"] + sar_metrics["f1"])
    mean_iou = 0.5 * (opt_metrics["iou"] + sar_metrics["iou"])
    return {
        "shift_dx": dx,
        "shift_dy": dy,
        "shift_l1": abs(dx) + abs(dy),
        "opt_oa": opt_metrics["oa"],
        "opt_precision": opt_metrics["precision"],
        "opt_recall": opt_metrics["recall"],
        "opt_f1": opt_metrics["f1"],
        "opt_iou": opt_metrics["iou"],
        "sar_oa": sar_metrics["oa"],
        "sar_precision": sar_metrics["precision"],
        "sar_recall": sar_metrics["recall"],
        "sar_f1": sar_metrics["f1"],
        "sar_iou": sar_metrics["iou"],
        "mean_f1": mean_f1,
        "mean_iou": mean_iou,
    }


def save_shift_summary(root, opt, dx, dy, opt_metrics, sar_metrics):
    os.makedirs(root, exist_ok=True)
    mean_f1 = 0.5 * (opt_metrics["f1"] + sar_metrics["f1"])
    mean_iou = 0.5 * (opt_metrics["iou"] + sar_metrics["iou"])
    path = os.path.join(root, "summary_metrics.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Experiment: {opt.exp_name}\n")
        f.write(f"Evaluation mode: {opt.mode}\n")
        if opt.mode == "full_scene":
            f.write(f"Scene: {opt.scene}\n")
        else:
            f.write(f"Dataset: {opt.dataset}\n")
        f.write(f"Artificial misalignment stage: {opt.misalignment_stage}\n")
        f.write(f"Misalignment branch: {opt.misalign_branch}\n")
        f.write(f"Input shift modality: {opt.input_shift_modality}\n")
        f.write(f"Valid-overlap evaluation: {opt.valid_overlap_only}\n")
        f.write(f"Shift dx: {dx}, dy: {dy}\n")
        f.write(f"Fill mode: {opt.shift_fill}\n\n")
        for name, metrics in (("Optical", opt_metrics), ("SAR", sar_metrics)):
            f.write(f"{name} CD result\n")
            f.write(
                f"TP: {metrics['tp']:.0f} TN: {metrics['tn']:.0f} "
                f"FP: {metrics['fp']:.0f} FN: {metrics['fn']:.0f}\n"
            )
            f.write(f"OA: {metrics['oa']:.6f}\n")
            f.write(f"Precision: {metrics['precision']:.6f}\n")
            f.write(f"Recall: {metrics['recall']:.6f}\n")
            f.write(f"F1: {metrics['f1']:.6f}\n")
            f.write(f"IoU: {metrics['iou']:.6f}\n\n")
        f.write(f"Mean F1: {mean_f1:.6f}\n")
        f.write(f"Mean IoU: {mean_iou:.6f}\n")
    return path


def save_example_panel(root, image1, image2, fake_opt, fake_sar, pred_opt, pred_sar, label, dx, dy):
    os.makedirs(root, exist_ok=True)
    save_visual_panel(
        [
            ("Image1 / O", to_uint8_img(image1)),
            ("Image2 / S", to_uint8_img(image2)),
            ("GT", cd_to_uint8(label)),
            (f"Gen optical shifted ({dx},{dy})", to_uint8_img(fake_opt)),
            (f"Gen SAR shifted ({dx},{dy})", to_uint8_img(fake_sar)),
            ("Opt CD", cd_to_uint8(pred_opt)),
            ("SAR CD", cd_to_uint8(pred_sar)),
        ],
        os.path.join(root, "misalignment_panel.png"),
        thumb_size=(256, 256),
        cols=4,
    )


def evaluate_patch_dataset(opt, device, models, dx, dy):
    G_opt2sar, G_sar2opt, D_opt, D_sar = models
    image_dir = os.path.join("dataset", opt.dataset, "test", "Image")
    image2_dir = os.path.join("dataset", opt.dataset, "test", "Image2")
    label_dir = os.path.join("dataset", opt.dataset, "test", "label")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Patch test directory not found: {image_dir}")

    image_names = sorted([x for x in os.listdir(image_dir) if is_image_file(x)])
    if opt.max_test_images is not None and opt.max_test_images > 0:
        image_names = image_names[: opt.max_test_images]
    image2_map = {os.path.splitext(x)[0]: x for x in os.listdir(image2_dir) if is_image_file(x)}
    label_map = {os.path.splitext(x)[0]: x for x in os.listdir(label_dir) if is_image_file(x)}

    opt_counts = {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
    sar_counts = {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
    example = None
    processed = 0

    with torch.no_grad():
        for image_name in image_names:
            stem = os.path.splitext(image_name)[0]
            image2_name = image2_map.get(stem)
            label_name = label_map.get(stem)
            if image2_name is None or label_name is None:
                continue

            image1 = np.array(Image.open(os.path.join(image_dir, image_name)).convert("RGB")).astype(np.float32) / 255.0
            image2 = np.array(Image.open(os.path.join(image2_dir, image2_name)).convert("RGB")).astype(np.float32) / 255.0
            label = np.array(Image.open(os.path.join(label_dir, label_name)).convert("L"))
            label = np.where(label > 0, 1, 0).astype(np.uint8)

            label_opt = label
            label_sar = label
            valid_mask = np.ones_like(label, dtype=np.uint8)
            if opt.misalignment_stage == "input":
                image1, image2, label_opt, label_sar, valid_mask = prepare_input_misalignment(
                    image1,
                    image2,
                    label,
                    dx,
                    dy,
                    opt.input_shift_modality,
                    opt.shift_fill,
                )

            real_A = np_to_model_tensor(image1, device, opt.input_value_range)
            real_B = np_to_model_tensor(image2, device, opt.input_value_range)
            fake_sar = G_opt2sar(real_A)
            fake_opt = G_sar2opt(real_B)
            if opt.misalignment_stage == "generated":
                fake_opt, fake_sar = maybe_shift_generated(
                    fake_opt, fake_sar, dx, dy, opt.misalign_branch, opt.shift_fill
                )
            _, _, out_opt = D_opt(real_A, fake_opt)
            _, _, out_sar = D_sar(real_B, fake_sar)
            pred_opt = torch.argmax(out_opt, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_sar = torch.argmax(out_sar, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            metric_mask = valid_mask if opt.valid_overlap_only and opt.misalignment_stage == "input" else None
            accumulate_counts(opt_counts, confusion_from_np(pred_opt, label_opt, metric_mask))
            accumulate_counts(sar_counts, confusion_from_np(pred_sar, label_sar, metric_mask))
            processed += 1

            if example is None and opt.save_visuals:
                example = {
                    "image1": image1,
                    "image2": image2,
                    "fake_opt": tensor_to_image_np(fake_opt, opt.generated_value_range),
                    "fake_sar": tensor_to_image_np(fake_sar, opt.generated_value_range),
                    "pred_opt": pred_opt,
                    "pred_sar": pred_sar,
                    "label": label,
                }

    opt_metrics = metrics_from_counts(opt_counts)
    sar_metrics = metrics_from_counts(sar_counts)
    return opt_metrics, sar_metrics, example, processed


def evaluate_full_scene(opt, device, models, dx, dy):
    if opt.scene not in SCENES:
        raise ValueError(f"Unknown scene: {opt.scene}. Available: {', '.join(sorted(SCENES))}")
    spec = SCENES[opt.scene]
    image1 = load_rgb_np(os.path.join(spec.source_dir, spec.image1))
    image2 = load_rgb_np(os.path.join(spec.source_dir, spec.image2))
    label = load_label_np(os.path.join(spec.source_dir, spec.label))
    if image1.shape[:2] != image2.shape[:2] or image1.shape[:2] != label.shape[:2]:
        raise ValueError("Scene image sizes are inconsistent.")

    label_opt = label
    label_sar = label
    valid_mask = np.ones_like(label, dtype=np.uint8)
    if opt.misalignment_stage == "input":
        image1, image2, label_opt, label_sar, valid_mask = prepare_input_misalignment(
            image1,
            image2,
            label,
            dx,
            dy,
            opt.input_shift_modality,
            opt.shift_fill,
        )

    image1, orig_h, orig_w = pad_to_patch(image1, opt.patch_size)
    image2, _, _ = pad_to_patch(image2, opt.patch_size)
    label, _, _ = pad_to_patch(label, opt.patch_size)
    label_opt, _, _ = pad_to_patch(label_opt, opt.patch_size)
    label_sar, _, _ = pad_to_patch(label_sar, opt.patch_size)
    valid_mask, _, _ = pad_to_patch(valid_mask, opt.patch_size)
    h, w = image1.shape[:2]
    xs = positions(w, opt.patch_size, opt.stride)
    ys = positions(h, opt.patch_size, opt.stride)
    print(
        f"Full scene size: original={orig_w}x{orig_h}, padded={w}x{h}, "
        f"windows={len(xs) * len(ys)}, shift=({dx},{dy})"
    )

    G_opt2sar, G_sar2opt, D_opt, D_sar = models
    prob_opt = np.zeros((2, h, w), dtype=np.float32)
    prob_sar = np.zeros((2, h, w), dtype=np.float32)
    fake_opt_full = np.zeros((h, w, 3), dtype=np.float32)
    fake_sar_full = np.zeros((h, w, 3), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    with torch.no_grad():
        idx = 0
        total = len(xs) * len(ys)
        for y in ys:
            for x in xs:
                idx += 1
                patch_a = image1[y : y + opt.patch_size, x : x + opt.patch_size]
                patch_b = image2[y : y + opt.patch_size, x : x + opt.patch_size]
                real_A = np_to_model_tensor(patch_a, device, opt.input_value_range)
                real_B = np_to_model_tensor(patch_b, device, opt.input_value_range)
                fake_sar = G_opt2sar(real_A)
                fake_opt = G_sar2opt(real_B)
                if opt.misalignment_stage == "generated":
                    fake_opt, fake_sar = maybe_shift_generated(
                        fake_opt, fake_sar, dx, dy, opt.misalign_branch, opt.shift_fill
                    )
                _, _, out_opt = D_opt(real_A, fake_opt)
                _, _, out_sar = D_sar(real_B, fake_sar)

                prob_opt[:, y : y + opt.patch_size, x : x + opt.patch_size] += F.softmax(
                    out_opt, dim=1
                ).squeeze(0).cpu().numpy()
                prob_sar[:, y : y + opt.patch_size, x : x + opt.patch_size] += F.softmax(
                    out_sar, dim=1
                ).squeeze(0).cpu().numpy()
                fake_opt_full[y : y + opt.patch_size, x : x + opt.patch_size] += tensor_to_image_np(
                    fake_opt, opt.generated_value_range
                )
                fake_sar_full[y : y + opt.patch_size, x : x + opt.patch_size] += tensor_to_image_np(
                    fake_sar, opt.generated_value_range
                )
                count[y : y + opt.patch_size, x : x + opt.patch_size] += 1.0
                if idx % 20 == 0 or idx == total:
                    print(f"  processed {idx}/{total} windows")

    count_safe = np.maximum(count, 1e-6)
    prob_opt = prob_opt / count_safe[None, :, :]
    prob_sar = prob_sar / count_safe[None, :, :]
    fake_opt_full = fake_opt_full / count_safe[:, :, None]
    fake_sar_full = fake_sar_full / count_safe[:, :, None]
    pred_opt = np.argmax(prob_opt, axis=0).astype(np.uint8)[:orig_h, :orig_w]
    pred_sar = np.argmax(prob_sar, axis=0).astype(np.uint8)[:orig_h, :orig_w]
    label_crop = label[:orig_h, :orig_w]
    label_opt_crop = label_opt[:orig_h, :orig_w]
    label_sar_crop = label_sar[:orig_h, :orig_w]
    valid_crop = valid_mask[:orig_h, :orig_w]
    metric_mask = valid_crop if opt.valid_overlap_only and opt.misalignment_stage == "input" else None
    opt_metrics = metrics_from_counts(confusion_from_np(pred_opt, label_opt_crop, metric_mask))
    sar_metrics = metrics_from_counts(confusion_from_np(pred_sar, label_sar_crop, metric_mask))
    example = None
    if opt.save_visuals:
        example = {
            "image1": image1[:orig_h, :orig_w],
            "image2": image2[:orig_h, :orig_w],
            "fake_opt": fake_opt_full[:orig_h, :orig_w],
            "fake_sar": fake_sar_full[:orig_h, :orig_w],
            "pred_opt": pred_opt,
            "pred_sar": pred_sar,
            "label": label_crop,
        }
    return opt_metrics, sar_metrics, example, len(xs) * len(ys)


def normalise_args(opt):
    opt.opt_cgdr_max_flow = float(resolve_optional(opt.opt_cgdr_max_flow, opt.cgdr_max_flow))
    opt.sar_cgdr_max_flow = float(resolve_optional(opt.sar_cgdr_max_flow, opt.cgdr_max_flow))
    opt.opt_cgdr_corr_threshold = float(resolve_optional(opt.opt_cgdr_corr_threshold, opt.cgdr_corr_threshold))
    opt.sar_cgdr_corr_threshold = float(resolve_optional(opt.sar_cgdr_corr_threshold, opt.cgdr_corr_threshold))
    opt.opt_cgdr_adaptive_gate_alpha = float(
        resolve_optional(opt.opt_cgdr_adaptive_gate_alpha, opt.cgdr_adaptive_gate_alpha)
    )
    opt.sar_cgdr_adaptive_gate_alpha = float(
        resolve_optional(opt.sar_cgdr_adaptive_gate_alpha, opt.cgdr_adaptive_gate_alpha)
    )
    opt.opt_cgdr_target_high_ratio = float(resolve_optional(opt.opt_cgdr_target_high_ratio, opt.cgdr_target_high_ratio))
    opt.sar_cgdr_target_high_ratio = float(resolve_optional(opt.sar_cgdr_target_high_ratio, opt.cgdr_target_high_ratio))
    opt.opt_cgdr_residual_suppress = float(resolve_optional(opt.opt_cgdr_residual_suppress, opt.cgdr_residual_suppress))
    opt.sar_cgdr_residual_suppress = float(resolve_optional(opt.sar_cgdr_residual_suppress, opt.cgdr_residual_suppress))
    opt.opt_cgdr_low_conf_flow_scale = float(
        resolve_optional(opt.opt_cgdr_low_conf_flow_scale, opt.cgdr_low_conf_flow_scale)
    )
    opt.sar_cgdr_low_conf_flow_scale = float(
        resolve_optional(opt.sar_cgdr_low_conf_flow_scale, opt.cgdr_low_conf_flow_scale)
    )
    opt.opt_cgdr_change_preserve_strength = float(
        resolve_optional(opt.opt_cgdr_change_preserve_strength, opt.cgdr_change_preserve_strength)
    )
    opt.sar_cgdr_change_preserve_strength = float(
        resolve_optional(opt.sar_cgdr_change_preserve_strength, opt.cgdr_change_preserve_strength)
    )
    opt.sar_lee_filter_mode = resolve_speckle_mode(opt.lee_filter_sar_cd, opt.sar_lee_filter_mode, default_when_true="both")
    return opt


def build_parser():
    parser = argparse.ArgumentParser(
        description="Artificial misalignment evaluation for HuiYan/CGDR change detection checkpoints."
    )
    parser.add_argument("--mode", type=str, default="patch", choices=["patch", "full_scene"])
    parser.add_argument("--dataset", type=str, default="Gloucester", help="Patch dataset name for mode=patch.")
    parser.add_argument("--scene", type=str, default="California", choices=sorted(SCENES), help="Scene name for mode=full_scene.")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max_test_images", type=int, default=None)
    parser.add_argument("--shift_list", type=str, default="0,0;2,0;4,0;8,0")
    parser.add_argument("--shift_fill", type=str, default="edge", choices=["edge", "zero"])
    parser.add_argument(
        "--misalignment_stage",
        type=str,
        default="generated",
        choices=["generated", "input"],
        help="inject shift into generated moving images or one original input modality",
    )
    parser.add_argument("--misalign_branch", type=str, default="both", choices=["both", "optical", "sar"])
    parser.add_argument(
        "--input_shift_modality",
        type=str,
        default="sar",
        choices=["optical", "sar"],
        help="original modality shifted when misalignment_stage=input",
    )
    parser.add_argument(
        "--valid_overlap_only",
        type=str2bool,
        default=True,
        help="exclude padded non-overlap borders from metrics for input-stage shifts",
    )
    parser.add_argument("--save_visuals", type=str2bool, default=True)
    parser.add_argument("--input_nc", type=int, default=3)
    parser.add_argument("--output_nc", type=int, default=3)
    parser.add_argument("--n_residual_blocks", type=int, default=9)
    parser.add_argument("--generator_type", type=str, default="huiyan", choices=["huiyan", "resnet"])
    parser.add_argument("--checkpoint_format", type=str, default="huiyan", choices=["huiyan", "baseline"])
    parser.add_argument("--input_value_range", type=str, default="neg_one_one", choices=["neg_one_one", "zero_one"])
    parser.add_argument("--generated_value_range", type=str, default="neg_one_one", choices=["neg_one_one", "zero_one"])
    parser.add_argument("--use_cgdr", type=str2bool, default=True)
    parser.add_argument("--cgdr_max_flow", type=float, default=6.0)
    parser.add_argument("--cgdr_corr_threshold", type=float, default=0.45)
    parser.add_argument("--cgdr_adaptive_gate_alpha", type=float, default=0.25)
    parser.add_argument("--cgdr_target_high_ratio", type=float, default=0.30)
    parser.add_argument("--cgdr_min_high_ratio", type=float, default=0.10)
    parser.add_argument("--cgdr_max_high_ratio", type=float, default=0.60)
    parser.add_argument("--cgdr_residual_suppress", type=float, default=0.35)
    parser.add_argument("--cgdr_low_conf_flow_scale", type=float, default=0.55)
    parser.add_argument("--cgdr_residual_conf_temperature", type=float, default=6.0)
    parser.add_argument("--cgdr_change_preserve_strength", type=float, default=0.0)
    parser.add_argument("--cgdr_change_preserve_threshold", type=float, default=0.35)
    parser.add_argument("--cgdr_change_preserve_temperature", type=float, default=10.0)
    parser.add_argument("--cgdr_use_speckle_filter", type=str2bool, default=True)
    parser.add_argument("--cgdr_use_coarse_fine_split", type=str2bool, default=True)
    parser.add_argument("--opt_cgdr_max_flow", type=float, default=None)
    parser.add_argument("--sar_cgdr_max_flow", type=float, default=None)
    parser.add_argument("--opt_cgdr_corr_threshold", type=float, default=None)
    parser.add_argument("--sar_cgdr_corr_threshold", type=float, default=None)
    parser.add_argument("--opt_cgdr_adaptive_gate_alpha", type=float, default=None)
    parser.add_argument("--sar_cgdr_adaptive_gate_alpha", type=float, default=None)
    parser.add_argument("--opt_cgdr_target_high_ratio", type=float, default=None)
    parser.add_argument("--sar_cgdr_target_high_ratio", type=float, default=None)
    parser.add_argument("--opt_cgdr_residual_suppress", type=float, default=None)
    parser.add_argument("--sar_cgdr_residual_suppress", type=float, default=None)
    parser.add_argument("--opt_cgdr_low_conf_flow_scale", type=float, default=None)
    parser.add_argument("--sar_cgdr_low_conf_flow_scale", type=float, default=None)
    parser.add_argument("--opt_cgdr_change_preserve_strength", type=float, default=None)
    parser.add_argument("--sar_cgdr_change_preserve_strength", type=float, default=None)
    parser.add_argument("--opt_cgdr_region_mode", type=str, default="correlation", choices=["correlation", "sar_scatter"])
    parser.add_argument("--sar_cgdr_region_mode", type=str, default="sar_scatter", choices=["correlation", "sar_scatter"])
    parser.add_argument("--cgdr_scatter_threshold", type=float, default=0.55)
    parser.add_argument("--lee_filter_sar_cd", type=str2bool, default=True)
    parser.add_argument(
        "--sar_lee_filter_mode",
        type=str,
        default="auto",
        choices=["auto", "none", "both", "fixed_only", "moving_only"],
    )
    parser.add_argument(
        "--opt_lee_filter_mode",
        type=str,
        default="none",
        choices=["none", "both", "fixed_only", "moving_only"],
    )
    return parser


def main():
    parser = build_parser()
    opt = normalise_args(parser.parse_args())
    shifts = parse_shift_list(opt.shift_list)

    if opt.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested but not available.")
    if torch.cuda.is_available():
        if opt.gpu_id < 0 or opt.gpu_id >= torch.cuda.device_count():
            raise ValueError(f"Invalid gpu_id={opt.gpu_id}, available GPUs: 0..{torch.cuda.device_count() - 1}")
        device = torch.device(f"cuda:{opt.gpu_id}")
        cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    print("Using experiment:", opt.exp_name)
    print("Artificial shifts:", shifts)
    print("Misalignment stage:", opt.misalignment_stage)
    if opt.misalignment_stage == "input":
        print("Shifted input modality:", opt.input_shift_modality)
        print("Valid-overlap evaluation:", opt.valid_overlap_only)
    else:
        print("Shifted generated branch:", opt.misalign_branch)

    models = build_models(opt, device)
    for model in models:
        model.eval()

    eval_name = opt.scene if opt.mode == "full_scene" else opt.dataset
    protocol_tag = opt.misalignment_stage
    if opt.misalignment_stage == "input":
        protocol_tag += f"_{opt.input_shift_modality}"
    else:
        protocol_tag += f"_{opt.misalign_branch}"
    out_root = os.path.join("result", opt.exp_name, f"artificial_misalignment_{protocol_tag}_{eval_name}")
    os.makedirs(out_root, exist_ok=True)

    rows = []
    for dx, dy in shifts:
        tag = shift_tag(dx, dy)
        shift_root = os.path.join(out_root, tag)
        print(f"\n===> Evaluating artificial shift {tag}")
        if opt.mode == "full_scene":
            opt_metrics, sar_metrics, example, processed = evaluate_full_scene(opt, device, models, dx, dy)
        else:
            opt_metrics, sar_metrics, example, processed = evaluate_patch_dataset(opt, device, models, dx, dy)

        save_shift_summary(shift_root, opt, dx, dy, opt_metrics, sar_metrics)
        if example is not None:
            save_example_panel(
                os.path.join(shift_root, "images"),
                example["image1"],
                example["image2"],
                example["fake_opt"],
                example["fake_sar"],
                example["pred_opt"],
                example["pred_sar"],
                example["label"],
                dx,
                dy,
            )
        row = row_from_metrics(dx, dy, opt_metrics, sar_metrics)
        row["processed_units"] = processed
        rows.append(row)
        print(
            f"{tag}: Opt-IoU={row['opt_iou']:.4f}, SAR-IoU={row['sar_iou']:.4f}, "
            f"Mean-IoU={row['mean_iou']:.4f}, Mean-F1={row['mean_f1']:.4f}"
        )

    csv_path = os.path.join(out_root, "misalignment_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    txt_path = os.path.join(out_root, "misalignment_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment: {opt.exp_name}\n")
        f.write(f"Evaluation mode: {opt.mode}\n")
        f.write(f"Evaluation target: {eval_name}\n")
        f.write(f"Misalignment stage: {opt.misalignment_stage}\n")
        f.write(f"Misalignment branch: {opt.misalign_branch}\n")
        f.write(f"Input shift modality: {opt.input_shift_modality}\n")
        f.write(f"Valid-overlap evaluation: {opt.valid_overlap_only}\n")
        f.write(f"Shift fill: {opt.shift_fill}\n\n")
        f.write("shift_dx\tshift_dy\tOpt-IoU\tSAR-IoU\tMean-IoU\tOpt-F1\tSAR-F1\tMean-F1\n")
        for row in rows:
            f.write(
                f"{row['shift_dx']}\t{row['shift_dy']}\t"
                f"{row['opt_iou'] * 100:.2f}\t{row['sar_iou'] * 100:.2f}\t{row['mean_iou'] * 100:.2f}\t"
                f"{row['opt_f1'] * 100:.2f}\t{row['sar_f1'] * 100:.2f}\t{row['mean_f1'] * 100:.2f}\n"
            )
    print(f"\nSaved artificial misalignment summary: {csv_path}")
    print(f"Saved paper-friendly table: {txt_path}")


if __name__ == "__main__":
    main()
