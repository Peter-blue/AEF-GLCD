from __future__ import print_function

import argparse
import csv
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from PIL import Image

from confusion_visualization import save_confusion_map, save_labeled_panel
from Model.Sun_Net_gan import Discriminator, GeneratorResNet
from models.generator import build_generator
from test import save_visual_panel, str2bool


@dataclass
class SceneSpec:
    source_dir: str
    image1: str
    image2: str
    label: str


SCENES = {
    "California": SceneSpec(
        source_dir=os.path.join("dataset", "California"),
        image1="California_t1.bmp",
        image2="California_t2.bmp",
        label="California_gt.bmp",
    ),
    "Shuguang": SceneSpec(
        source_dir=os.path.join("dataset", "Shuguang"),
        image1="shuguang_1.bmp",
        image2="shuguang_2.bmp",
        label="shuguang_gt.bmp",
    ),
}


def positions(length, patch_size, stride):
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def pad_to_patch(arr, patch_size):
    h, w = arr.shape[:2]
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    if pad_h == 0 and pad_w == 0:
        return arr, h, w
    if arr.ndim == 3:
        padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    else:
        padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="edge")
    return padded, h, w


def load_rgb_np(path):
    return np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def load_label_np(path):
    arr = np.array(Image.open(path).convert("L"))
    return np.where(arr > 0, 1, 0).astype(np.uint8)


def np_to_model_tensor(patch, device, value_range):
    patch = np.transpose(patch, (2, 0, 1))
    tensor = torch.tensor(patch, device=device, dtype=torch.float32).unsqueeze(0)
    if value_range == "neg_one_one":
        return tensor * 2.0 - 1.0
    if value_range == "zero_one":
        return tensor
    raise ValueError(f"Unknown input_value_range: {value_range}")


def tensor_to_image_np(x, value_range):
    x = x.detach().squeeze(0).float().cpu().clamp(-1.0, 1.0)
    if value_range == "neg_one_one":
        x = ((x + 1.0) * 0.5).numpy()
    elif value_range == "zero_one":
        x = x.clamp(0.0, 1.0).numpy()
    else:
        raise ValueError(f"Unknown generated_value_range: {value_range}")
    return np.transpose(x, (1, 2, 0))


def to_uint8_img(x):
    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def cd_to_uint8(x):
    return np.where(x > 0, 255, 0).astype(np.uint8)


def compute_metrics(tp, tn, fp, fn, eps=1e-8):
    oa = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return oa, precision, recall, f1, iou


def metric_row(name, pred, label):
    pred = pred.astype(np.uint8)
    label = label.astype(np.uint8)
    tp = float(np.logical_and(pred == 1, label == 1).sum())
    tn = float(np.logical_and(pred == 0, label == 0).sum())
    fp = float(np.logical_and(pred == 1, label == 0).sum())
    fn = float(np.logical_and(pred == 0, label == 1).sum())
    oa, precision, recall, f1, iou = compute_metrics(tp, tn, fp, fn)
    return {
        "branch": name,
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


def resolve_optional(value, fallback):
    return fallback if value is None else value


def resolve_speckle_mode(legacy_flag, mode, default_when_true="both"):
    if mode is None or mode == "auto":
        return default_when_true if legacy_flag else "none"
    return mode


def build_models(opt, device):
    input_shape = (opt.input_nc, opt.patch_size, opt.patch_size)
    if opt.generator_type == "resnet":
        G_opt2sar = GeneratorResNet(input_shape, opt.n_residual_blocks)
        G_sar2opt = GeneratorResNet(input_shape, opt.n_residual_blocks)
    elif opt.generator_type == "huiyan":
        G_opt2sar = build_generator(opt.input_nc, opt.output_nc, opt.n_residual_blocks)
        G_sar2opt = build_generator(opt.input_nc, opt.output_nc, opt.n_residual_blocks)
    else:
        raise ValueError(f"Unknown generator_type: {opt.generator_type}")
    D_opt = Discriminator(
        input_shape,
        use_cgdr=opt.use_cgdr,
        cgdr_max_flow=opt.opt_cgdr_max_flow,
        cgdr_corr_threshold=opt.opt_cgdr_corr_threshold,
        cgdr_region_mode=opt.opt_cgdr_region_mode,
        cgdr_scatter_threshold=opt.cgdr_scatter_threshold,
        cgdr_adaptive_gate_alpha=opt.opt_cgdr_adaptive_gate_alpha,
        cgdr_target_high_ratio=opt.opt_cgdr_target_high_ratio,
        cgdr_min_high_ratio=opt.cgdr_min_high_ratio,
        cgdr_max_high_ratio=opt.cgdr_max_high_ratio,
        cgdr_residual_suppress=opt.opt_cgdr_residual_suppress,
        cgdr_low_conf_flow_scale=opt.opt_cgdr_low_conf_flow_scale,
        cgdr_residual_conf_temperature=opt.cgdr_residual_conf_temperature,
        cgdr_mask_alignment_with_valid=True,
        cgdr_change_preserve_strength=opt.opt_cgdr_change_preserve_strength,
        cgdr_change_preserve_threshold=opt.cgdr_change_preserve_threshold,
        cgdr_change_preserve_temperature=opt.cgdr_change_preserve_temperature,
        cgdr_use_speckle_filter=opt.cgdr_use_speckle_filter,
        cgdr_use_coarse_fine_split=opt.cgdr_use_coarse_fine_split,
        filter_speckle_for_cd=False,
        speckle_filter_mode=opt.opt_lee_filter_mode,
    )
    D_sar = Discriminator(
        input_shape,
        use_cgdr=opt.use_cgdr,
        cgdr_max_flow=opt.sar_cgdr_max_flow,
        cgdr_corr_threshold=opt.sar_cgdr_corr_threshold,
        cgdr_region_mode=opt.sar_cgdr_region_mode,
        cgdr_scatter_threshold=opt.cgdr_scatter_threshold,
        cgdr_adaptive_gate_alpha=opt.sar_cgdr_adaptive_gate_alpha,
        cgdr_target_high_ratio=opt.sar_cgdr_target_high_ratio,
        cgdr_min_high_ratio=opt.cgdr_min_high_ratio,
        cgdr_max_high_ratio=opt.cgdr_max_high_ratio,
        cgdr_residual_suppress=opt.sar_cgdr_residual_suppress,
        cgdr_low_conf_flow_scale=opt.sar_cgdr_low_conf_flow_scale,
        cgdr_residual_conf_temperature=opt.cgdr_residual_conf_temperature,
        cgdr_mask_alignment_with_valid=True,
        cgdr_change_preserve_strength=opt.sar_cgdr_change_preserve_strength,
        cgdr_change_preserve_threshold=opt.cgdr_change_preserve_threshold,
        cgdr_change_preserve_temperature=opt.cgdr_change_preserve_temperature,
        cgdr_use_speckle_filter=opt.cgdr_use_speckle_filter,
        cgdr_use_coarse_fine_split=opt.cgdr_use_coarse_fine_split,
        filter_speckle_for_cd=opt.lee_filter_sar_cd,
        speckle_filter_mode=opt.sar_lee_filter_mode,
    )

    ckpt_dir = os.path.join("checkpoint", opt.exp_name)
    if opt.checkpoint_format == "huiyan":
        names = {
            "G_opt2sar": "G_opt2sar_best.pth",
            "G_sar2opt": "G_sar2opt_best.pth",
            "D_opt": "D_opt_best.pth",
            "D_sar": "D_sar_best.pth",
        }
    elif opt.checkpoint_format == "baseline":
        names = {
            "G_opt2sar": "G_AB_best.pth",
            "G_sar2opt": "G_BA_best.pth",
            "D_opt": "D_A_best.pth",
            "D_sar": "D_B_best.pth",
        }
    else:
        raise ValueError(f"Unknown checkpoint_format: {opt.checkpoint_format}")

    G_opt2sar.load_state_dict(torch.load(os.path.join(ckpt_dir, names["G_opt2sar"]), map_location="cpu"))
    G_sar2opt.load_state_dict(torch.load(os.path.join(ckpt_dir, names["G_sar2opt"]), map_location="cpu"))
    missing_opt, unexpected_opt = D_opt.load_state_dict(
        torch.load(os.path.join(ckpt_dir, names["D_opt"]), map_location="cpu"), strict=False
    )
    missing_sar, unexpected_sar = D_sar.load_state_dict(
        torch.load(os.path.join(ckpt_dir, names["D_sar"]), map_location="cpu"), strict=False
    )
    if missing_opt or unexpected_opt:
        print(f"[WARN] D_opt mismatch: missing={len(missing_opt)}, unexpected={len(unexpected_opt)}")
    if missing_sar or unexpected_sar:
        print(f"[WARN] D_sar mismatch: missing={len(missing_sar)}, unexpected={len(unexpected_sar)}")

    models = [G_opt2sar, G_sar2opt, D_opt, D_sar]
    models = [m.to(device).eval() for m in models]
    return models


def sliding_full_scene_inference(opt, device):
    if opt.scene not in SCENES:
        raise ValueError(f"Unknown scene: {opt.scene}. Available: {', '.join(sorted(SCENES))}")
    spec = SCENES[opt.scene]
    image1 = load_rgb_np(os.path.join(spec.source_dir, spec.image1))
    image2 = load_rgb_np(os.path.join(spec.source_dir, spec.image2))
    label = load_label_np(os.path.join(spec.source_dir, spec.label))
    if image1.shape[:2] != image2.shape[:2] or image1.shape[:2] != label.shape[:2]:
        raise ValueError("Scene image sizes are inconsistent.")

    image1, orig_h, orig_w = pad_to_patch(image1, opt.patch_size)
    image2, _, _ = pad_to_patch(image2, opt.patch_size)
    label, _, _ = pad_to_patch(label, opt.patch_size)
    h, w = image1.shape[:2]
    xs = positions(w, opt.patch_size, opt.stride)
    ys = positions(h, opt.patch_size, opt.stride)
    print(f"Full scene size: original={orig_w}x{orig_h}, padded={w}x{h}, windows={len(xs) * len(ys)}")

    G_opt2sar, G_sar2opt, D_opt, D_sar = build_models(opt, device)
    prob_opt = np.zeros((2, h, w), dtype=np.float32)
    prob_sar = np.zeros((2, h, w), dtype=np.float32)
    fake_opt = np.zeros((h, w, 3), dtype=np.float32)
    fake_sar = np.zeros((h, w, 3), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    with torch.no_grad():
        idx = 0
        for y in ys:
            for x in xs:
                idx += 1
                patch_a = image1[y : y + opt.patch_size, x : x + opt.patch_size]
                patch_b = image2[y : y + opt.patch_size, x : x + opt.patch_size]
                real_A = np_to_model_tensor(patch_a, device, opt.input_value_range)
                real_B = np_to_model_tensor(patch_b, device, opt.input_value_range)
                gen_sar = G_opt2sar(real_A)
                gen_opt = G_sar2opt(real_B) 
                _, _, out_opt = D_opt(real_A, gen_opt)
                _, _, out_sar = D_sar(real_B, gen_sar)
                p_opt = F.softmax(out_opt, dim=1).squeeze(0).cpu().numpy()
                p_sar = F.softmax(out_sar, dim=1).squeeze(0).cpu().numpy()
                prob_opt[:, y : y + opt.patch_size, x : x + opt.patch_size] += p_opt
                prob_sar[:, y : y + opt.patch_size, x : x + opt.patch_size] += p_sar
                fake_opt[y : y + opt.patch_size, x : x + opt.patch_size] += tensor_to_image_np(
                    gen_opt, opt.generated_value_range
                )
                fake_sar[y : y + opt.patch_size, x : x + opt.patch_size] += tensor_to_image_np(
                    gen_sar, opt.generated_value_range
                )
                count[y : y + opt.patch_size, x : x + opt.patch_size] += 1.0
                if idx % 20 == 0 or idx == len(xs) * len(ys):
                    print(f"  processed {idx}/{len(xs) * len(ys)} windows")

    count_safe = np.maximum(count, 1e-6)
    prob_opt = prob_opt / count_safe[None, :, :]
    prob_sar = prob_sar / count_safe[None, :, :]
    fake_opt = fake_opt / count_safe[:, :, None]
    fake_sar = fake_sar / count_safe[:, :, None]
    pred_opt = np.argmax(prob_opt, axis=0).astype(np.uint8)[:orig_h, :orig_w]
    pred_sar = np.argmax(prob_sar, axis=0).astype(np.uint8)[:orig_h, :orig_w]
    return {
        "image1": image1[:orig_h, :orig_w],
        "image2": image2[:orig_h, :orig_w],
        "label": label[:orig_h, :orig_w],
        "fake_opt": fake_opt[:orig_h, :orig_w],
        "fake_sar": fake_sar[:orig_h, :orig_w],
        "pred_opt": pred_opt,
        "pred_sar": pred_sar,
        "prob_opt_change": prob_opt[1, :orig_h, :orig_w],
        "prob_sar_change": prob_sar[1, :orig_h, :orig_w],
    }


def save_outputs(opt, outputs):
    out_dir = os.path.join("result", opt.exp_name, f"full_scene_{opt.scene}")
    metrics_dir = os.path.join(out_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    Image.fromarray(to_uint8_img(outputs["image1"])).save(os.path.join(out_dir, "images", "real_image1.png"))
    Image.fromarray(to_uint8_img(outputs["image2"])).save(os.path.join(out_dir, "images", "real_image2.png"))
    Image.fromarray(to_uint8_img(outputs["fake_opt"])).save(os.path.join(out_dir, "images", "generated_optical_full.png"))
    Image.fromarray(to_uint8_img(outputs["fake_sar"])).save(os.path.join(out_dir, "images", "generated_sar_full.png"))
    Image.fromarray(cd_to_uint8(outputs["label"])).save(os.path.join(out_dir, "images", "label_full.png"))
    Image.fromarray(cd_to_uint8(outputs["pred_opt"])).save(os.path.join(out_dir, "images", "optical_cd_full.png"))
    Image.fromarray(cd_to_uint8(outputs["pred_sar"])).save(os.path.join(out_dir, "images", "sar_cd_full.png"))
    opt_confusion = os.path.join(out_dir, "images", "optical_confusion_map.png")
    sar_confusion = os.path.join(out_dir, "images", "sar_confusion_map.png")
    save_confusion_map(outputs["pred_opt"], outputs["label"], opt_confusion)
    save_confusion_map(outputs["pred_sar"], outputs["label"], sar_confusion)
    save_labeled_panel(
        [
            ("Ground Truth", Image.fromarray(cd_to_uint8(outputs["label"])).convert("RGB")),
            ("Optical branch", Image.open(opt_confusion).convert("RGB")),
            ("SAR branch", Image.open(sar_confusion).convert("RGB")),
        ],
        os.path.join(out_dir, "confusion_panel.png"),
        tile_size=(320, 320),
        title=f"{opt.scene} - TP/TN/FP/FN",
    )
    Image.fromarray(to_uint8_img(outputs["prob_opt_change"])).save(os.path.join(out_dir, "images", "optical_change_prob.png"))
    Image.fromarray(to_uint8_img(outputs["prob_sar_change"])).save(os.path.join(out_dir, "images", "sar_change_prob.png"))
    save_visual_panel(
        [
            ("Image1 / O", to_uint8_img(outputs["image1"])),
            ("Image2 / S", to_uint8_img(outputs["image2"])),
            ("GT", cd_to_uint8(outputs["label"])),
            ("Gen optical", to_uint8_img(outputs["fake_opt"])),
            ("Gen SAR", to_uint8_img(outputs["fake_sar"])),
            ("Opt CD", cd_to_uint8(outputs["pred_opt"])),
            ("SAR CD", cd_to_uint8(outputs["pred_sar"])),
        ],
        os.path.join(out_dir, "full_scene_panel.png"),
        thumb_size=(256, 256),
        cols=4,
    )

    rows = [
        metric_row("Optical", outputs["pred_opt"], outputs["label"]),
        metric_row("SAR", outputs["pred_sar"], outputs["label"]),
    ]
    mean_iou = 0.5 * (rows[0]["iou"] + rows[1]["iou"])
    mean_f1 = 0.5 * (rows[0]["f1"] + rows[1]["f1"])
    with open(os.path.join(metrics_dir, "summary_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Experiment: {opt.exp_name}\n")
        f.write(f"Scene: {opt.scene}\n")
        f.write(f"Patch size: {opt.patch_size}, stride: {opt.stride}\n\n")
        for row in rows:
            f.write(f"{row['branch']} CD result\n")
            f.write(f"TP: {row['tp']:.0f} TN: {row['tn']:.0f} FP: {row['fp']:.0f} FN: {row['fn']:.0f}\n")
            f.write(f"OA: {row['oa']:.6f}\n")
            f.write(f"Precision: {row['precision']:.6f}\n")
            f.write(f"Recall: {row['recall']:.6f}\n")
            f.write(f"F1: {row['f1']:.6f}\n")
            f.write(f"IoU: {row['iou']:.6f}\n\n")
        f.write(f"Mean F1: {mean_f1:.6f}\n")
        f.write(f"Mean IoU: {mean_iou:.6f}\n")

    with open(os.path.join(metrics_dir, "summary_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved full-scene results: {out_dir}")
    print(f"Mean F1={mean_f1:.6f}, Mean IoU={mean_iou:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Sliding-window full-scene inference for single-scene CD datasets.")
    parser.add_argument("--scene", required=True, choices=sorted(SCENES))
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
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
    parser.add_argument("--sar_lee_filter_mode", type=str, default="auto", choices=["auto", "none", "both", "fixed_only", "moving_only"])
    parser.add_argument("--opt_lee_filter_mode", type=str, default="none", choices=["none", "both", "fixed_only", "moving_only"])
    opt = parser.parse_args()

    opt.opt_cgdr_max_flow = float(resolve_optional(opt.opt_cgdr_max_flow, opt.cgdr_max_flow))
    opt.sar_cgdr_max_flow = float(resolve_optional(opt.sar_cgdr_max_flow, opt.cgdr_max_flow))
    opt.opt_cgdr_corr_threshold = float(resolve_optional(opt.opt_cgdr_corr_threshold, opt.cgdr_corr_threshold))
    opt.sar_cgdr_corr_threshold = float(resolve_optional(opt.sar_cgdr_corr_threshold, opt.cgdr_corr_threshold))
    opt.opt_cgdr_adaptive_gate_alpha = float(resolve_optional(opt.opt_cgdr_adaptive_gate_alpha, opt.cgdr_adaptive_gate_alpha))
    opt.sar_cgdr_adaptive_gate_alpha = float(resolve_optional(opt.sar_cgdr_adaptive_gate_alpha, opt.cgdr_adaptive_gate_alpha))
    opt.opt_cgdr_target_high_ratio = float(resolve_optional(opt.opt_cgdr_target_high_ratio, opt.cgdr_target_high_ratio))
    opt.sar_cgdr_target_high_ratio = float(resolve_optional(opt.sar_cgdr_target_high_ratio, opt.cgdr_target_high_ratio))
    opt.opt_cgdr_residual_suppress = float(resolve_optional(opt.opt_cgdr_residual_suppress, opt.cgdr_residual_suppress))
    opt.sar_cgdr_residual_suppress = float(resolve_optional(opt.sar_cgdr_residual_suppress, opt.cgdr_residual_suppress))
    opt.opt_cgdr_low_conf_flow_scale = float(resolve_optional(opt.opt_cgdr_low_conf_flow_scale, opt.cgdr_low_conf_flow_scale))
    opt.sar_cgdr_low_conf_flow_scale = float(resolve_optional(opt.sar_cgdr_low_conf_flow_scale, opt.cgdr_low_conf_flow_scale))
    opt.opt_cgdr_change_preserve_strength = float(resolve_optional(opt.opt_cgdr_change_preserve_strength, opt.cgdr_change_preserve_strength))
    opt.sar_cgdr_change_preserve_strength = float(resolve_optional(opt.sar_cgdr_change_preserve_strength, opt.cgdr_change_preserve_strength))
    opt.sar_lee_filter_mode = resolve_speckle_mode(opt.lee_filter_sar_cd, opt.sar_lee_filter_mode, default_when_true="both")

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
    print(opt)

    outputs = sliding_full_scene_inference(opt, device)
    save_outputs(opt, outputs)


if __name__ == "__main__":
    main()
