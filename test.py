from __future__ import print_function

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image, ImageDraw

from confusion_visualization import save_confusion_map
from Model.Sun_Net_gan import Discriminator
from models.generator import build_generator
from utils import is_image_file


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y", "t")


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def compute_metrics(tp, tn, fp, fn, eps=1e-8):
    oa = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return oa, precision, recall, f1, iou


def confusion_counts(pred, lbl):
    tp = ((pred == 1).long() & (lbl == 1).long()).float().sum().item()
    tn = ((pred == 0).long() & (lbl == 0).long()).float().sum().item()
    fp = ((pred == 1).long() & (lbl == 0).long()).float().sum().item()
    fn = ((pred == 0).long() & (lbl == 1).long()).float().sum().item()
    return tp, tn, fp, fn


def metric_dict(tp, tn, fp, fn):
    oa, precision, recall, f1, iou = compute_metrics(tp, tn, fp, fn)
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


def tensor_to_uint8_img(x):
    # x in [-1, 1] -> [0, 255]
    x = x.detach().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 0.5 * 255.0).squeeze(0).cpu().numpy().astype(np.uint8)
    return np.transpose(x, (1, 2, 0))


def tensor_to_gray_img(x, value_range="auto"):
    x = x.detach().squeeze().float().cpu().numpy()
    if x.ndim == 3:
        x = x[0]
    if value_range == "corr":
        x = (x + 1.0) * 0.5
    elif value_range == "auto":
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        x = (x - x_min) / (x_max - x_min + 1e-8)
    else:
        x = np.clip(x, 0.0, 1.0)
    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def hsv_to_rgb_np(h, s, v):
    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i_mod = i % 6
    rgb = np.zeros(h.shape + (3,), dtype=np.float32)
    masks = [
        (i_mod == 0, np.stack([v, t, p], axis=-1)),
        (i_mod == 1, np.stack([q, v, p], axis=-1)),
        (i_mod == 2, np.stack([p, v, t], axis=-1)),
        (i_mod == 3, np.stack([p, q, v], axis=-1)),
        (i_mod == 4, np.stack([t, p, v], axis=-1)),
        (i_mod == 5, np.stack([v, p, q], axis=-1)),
    ]
    for mask, values in masks:
        rgb[mask] = values[mask]
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def flow_to_rgb_img(flow):
    flow = flow.detach().squeeze(0).float().cpu().numpy()
    u = flow[0]
    v = flow[1]
    mag = np.sqrt(u * u + v * v)
    angle = np.arctan2(v, u)
    hue = (angle + np.pi) / (2.0 * np.pi)
    value = mag / (np.percentile(mag, 95) + 1e-8)
    value = np.clip(value, 0.0, 1.0)
    sat = np.ones_like(value, dtype=np.float32)
    return hsv_to_rgb_np(hue.astype(np.float32), sat, value.astype(np.float32))


def ensure_rgb_img(img):
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 1:
        return np.repeat(img, 3, axis=-1)
    return img


def save_visual_panel(items, path, thumb_size=(256, 256), cols=3):
    """Save a compact labeled panel for paper-friendly CGDR diagnostics."""
    if not items:
        return
    cols = max(1, int(cols))
    rows = int(np.ceil(len(items) / float(cols)))
    thumb_w, thumb_h = thumb_size
    label_h = 24
    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for idx, (title, img) in enumerate(items):
        row = idx // cols
        col = idx % cols
        x0 = col * thumb_w
        y0 = row * (thumb_h + label_h)
        img = ensure_rgb_img(img).astype(np.uint8)
        resample = Image.NEAREST if img.ndim == 2 else Image.BILINEAR
        tile = Image.fromarray(img).resize((thumb_w, thumb_h), resample=resample)
        canvas.paste(tile, (x0, y0 + label_h))
        draw.rectangle([x0, y0, x0 + thumb_w, y0 + label_h], fill=(245, 245, 245))
        draw.text((x0 + 6, y0 + 5), str(title), fill=(20, 20, 20))

    canvas.save(path)


def save_cd_map(pred, path):
    cd = pred.detach().cpu().numpy().astype(np.uint8)
    cd = np.where(cd > 0, 255, 0).astype(np.uint8)
    Image.fromarray(cd).save(path)


def save_cgdr_branch_visuals(root, branch_name, fixed, moving, pred, aux, scatter_threshold):
    branch_dir = os.path.join(root, branch_name)
    os.makedirs(branch_dir, exist_ok=True)

    fixed_img = tensor_to_uint8_img(fixed)
    moving_img = tensor_to_uint8_img(moving)
    aligned = aux.get("aligned", moving) if aux is not None else moving
    aligned_img = tensor_to_uint8_img(aligned)
    cd_img = pred.detach().cpu().numpy().astype(np.uint8)
    cd_img = np.where(cd_img > 0, 255, 0).astype(np.uint8)

    Image.fromarray(fixed_img).save(os.path.join(branch_dir, "fixed_real.png"))
    Image.fromarray(moving_img).save(os.path.join(branch_dir, "moving_generated_before.png"))
    Image.fromarray(aligned_img).save(os.path.join(branch_dir, "moving_generated_after.png"))
    before_pair = np.concatenate([fixed_img, moving_img], axis=1)
    after_pair = np.concatenate([fixed_img, aligned_img], axis=1)
    Image.fromarray(before_pair).save(os.path.join(branch_dir, "before_alignment.png"))
    Image.fromarray(after_pair).save(os.path.join(branch_dir, "after_alignment.png"))
    save_cd_map(pred, os.path.join(branch_dir, "final_cd_map.png"))

    blank_gray = np.zeros(fixed_img.shape[:2], dtype=np.uint8)
    blank_rgb = np.zeros_like(fixed_img)
    corr_img = blank_gray
    high_img = blank_gray
    scatter_img = blank_gray
    flow_img = blank_rgb

    if aux is None:
        Image.fromarray(corr_img).save(os.path.join(branch_dir, "correlation_map.png"))
        Image.fromarray(high_img).save(os.path.join(branch_dir, "high_correlation_mask.png"))
        Image.fromarray(scatter_img).save(os.path.join(branch_dir, "strong_scatter_mask.png"))
        Image.fromarray(flow_img).save(os.path.join(branch_dir, "flow_field.png"))
        save_visual_panel(
            [
                ("fixed real", fixed_img),
                ("moving before", moving_img),
                ("moving after", aligned_img),
                ("correlation", corr_img),
                ("high mask", high_img),
                ("scatter mask", scatter_img),
                ("flow field", flow_img),
                ("CD map", cd_img),
            ],
            os.path.join(branch_dir, "cgdr_alignment_panel.png"),
            cols=4,
        )
        return

    corr = aux.get("corr_after", aux.get("corr_fine", None))
    if corr is not None:
        corr_img = tensor_to_gray_img(corr, value_range="corr")
    Image.fromarray(corr_img).save(os.path.join(branch_dir, "correlation_map.png"))

    high_gate = aux.get("high_gate", None)
    if high_gate is not None:
        high_img = tensor_to_gray_img(high_gate, value_range="unit")
    Image.fromarray(high_img).save(os.path.join(branch_dir, "high_correlation_mask.png"))

    scatter = aux.get("scatter_score", None)
    if scatter is not None:
        scatter_img = (scatter.detach().squeeze().float().cpu().numpy() >= float(scatter_threshold)).astype(np.uint8) * 255
    Image.fromarray(scatter_img).save(os.path.join(branch_dir, "strong_scatter_mask.png"))

    flow = aux.get("flow", None)
    if flow is not None:
        flow_img = flow_to_rgb_img(flow)
    Image.fromarray(flow_img).save(os.path.join(branch_dir, "flow_field.png"))
    save_visual_panel(
        [
            ("fixed real", fixed_img),
            ("moving before", moving_img),
            ("moving after", aligned_img),
            ("correlation", corr_img),
            ("high mask", high_img),
            ("scatter mask", scatter_img),
            ("flow field", flow_img),
            ("CD map", cd_img),
        ],
        os.path.join(branch_dir, "cgdr_alignment_panel.png"),
        cols=4,
    )


def to_neg_one_pos_one(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def resolve_optional(value, fallback):
    return fallback if value is None else value


def resolve_speckle_mode(legacy_flag, mode, default_when_true="both"):
    if mode is None or mode == "auto":
        return default_when_true if legacy_flag else "none"
    return mode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MTCDN + HuiYan generator inference")
    parser.add_argument("--dataset", required=False, default="Gloucester")
    parser.add_argument("--batch_size", type=int, default=1, help="training batch size")
    parser.add_argument("--test_batch_size", type=int, default=2, help="testing batch size")
    parser.add_argument("--direction", type=str, default="a2b", help="a2b or b2a")
    parser.add_argument("--input_nc", type=int, default=3, help="input image channels")
    parser.add_argument("--output_nc", type=int, default=3, help="output image channels")
    parser.add_argument("--ngf", type=int, default=64, help="generator filters in first conv layer")
    parser.add_argument("--ndf", type=int, default=512, help="discriminator filters in first conv layer")
    parser.add_argument("--epoch", type=int, default=0, help="legacy arg")
    parser.add_argument("--exp_name", type=str, default=None, help="explicit experiment folder under checkpoint/")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id")
    parser.add_argument("--threads", type=int, default=4, help="legacy arg")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--lamb", type=int, default=10, help="legacy arg")
    parser.add_argument("--img_height", type=int, default=256, help="image height")
    parser.add_argument("--img_width", type=int, default=256, help="image width")
    parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
    parser.add_argument("--lambda_cyc", type=float, default=10.0, help="legacy arg")
    parser.add_argument("--lambda_id", type=float, default=5.0, help="legacy arg")
    parser.add_argument("--freeze_generator_layers", type=int, default=10, help="legacy arg for compatibility")
    parser.add_argument("--save_generated_images", type=str2bool, default=True, help="save generated SAR images")
    parser.add_argument("--use_cgdr", type=str2bool, default=True, help="enable correlation-guided deformable registration")
    parser.add_argument("--use_ucef", type=str2bool, default=False, help="enable uncertainty-aware change evidence fusion in SNUNet_ECAM")
    parser.add_argument("--ucef_scale", type=float, default=0.5, help="residual scale of UCEF evidence enhancement")
    parser.add_argument("--use_racr", type=str2bool, default=False, help="enable registration-aware change refinement in SNUNet_ECAM")
    parser.add_argument("--racr_scale", type=float, default=0.2, help="learnable logit-refinement scale for RACR")
    parser.add_argument("--racr_base_suppress", type=float, default=0.05, help="conservative base suppression for unreliable change responses")
    parser.add_argument("--cgdr_max_flow", type=float, default=6.0, help="maximum coarse CGDR displacement in pixels")
    parser.add_argument("--cgdr_corr_threshold", type=float, default=0.45, help="hard fine-alignment threshold for CGDR")
    parser.add_argument("--cgdr_adaptive_gate_alpha", type=float, default=0.25, help="blend ratio for adaptive gate threshold")
    parser.add_argument("--cgdr_target_high_ratio", type=float, default=0.30, help="target ratio for high-confidence fine alignment")
    parser.add_argument("--cgdr_min_high_ratio", type=float, default=0.10, help="minimum allowed high-confidence ratio")
    parser.add_argument("--cgdr_max_high_ratio", type=float, default=0.60, help="maximum allowed high-confidence ratio")
    parser.add_argument("--cgdr_residual_suppress", type=float, default=0.35, help="suppress fine alignment in high-residual regions")
    parser.add_argument("--cgdr_low_conf_flow_scale", type=float, default=0.55, help="scale coarse flow in low-confidence regions")
    parser.add_argument("--cgdr_residual_conf_temperature", type=float, default=6.0, help="residual-to-confidence temperature")
    parser.add_argument("--cgdr_mask_alignment_with_valid", type=str2bool, default=True, help="compatibility arg; valid masks are only available in training")
    parser.add_argument("--cgdr_change_preserve_strength", type=float, default=0.0, help="reduce warping in high-residual change-like regions")
    parser.add_argument("--cgdr_change_preserve_threshold", type=float, default=0.35, help="residual threshold for change-preserving gate")
    parser.add_argument("--cgdr_change_preserve_temperature", type=float, default=10.0, help="temperature for change-preserving gate")
    parser.add_argument("--cgdr_use_speckle_filter", type=str2bool, default=True, help="use Lee-style speckle suppression inside CGDR")
    parser.add_argument("--cgdr_use_coarse_fine_split", type=str2bool, default=True, help="use CGDR coarse/fine region split with fine residual flow")
    parser.add_argument("--opt_cgdr_max_flow", type=float, default=None, help="optional SAR->Optical branch max coarse flow override")
    parser.add_argument("--sar_cgdr_max_flow", type=float, default=None, help="optional Optical->SAR branch max coarse flow override")
    parser.add_argument("--opt_cgdr_corr_threshold", type=float, default=None, help="optional SAR->Optical high-correlation threshold override")
    parser.add_argument("--sar_cgdr_corr_threshold", type=float, default=None, help="optional Optical->SAR high-correlation threshold override")
    parser.add_argument("--opt_cgdr_adaptive_gate_alpha", type=float, default=None, help="optional SAR->Optical adaptive gate alpha override")
    parser.add_argument("--sar_cgdr_adaptive_gate_alpha", type=float, default=None, help="optional Optical->SAR adaptive gate alpha override")
    parser.add_argument("--opt_cgdr_target_high_ratio", type=float, default=None, help="optional SAR->Optical target high-ratio override")
    parser.add_argument("--sar_cgdr_target_high_ratio", type=float, default=None, help="optional Optical->SAR target high-ratio override")
    parser.add_argument("--opt_cgdr_residual_suppress", type=float, default=None, help="optional SAR->Optical residual suppression override")
    parser.add_argument("--sar_cgdr_residual_suppress", type=float, default=None, help="optional Optical->SAR residual suppression override")
    parser.add_argument("--opt_cgdr_low_conf_flow_scale", type=float, default=None, help="optional SAR->Optical low-confidence flow-scale override")
    parser.add_argument("--sar_cgdr_low_conf_flow_scale", type=float, default=None, help="optional Optical->SAR low-confidence flow-scale override")
    parser.add_argument("--opt_cgdr_change_preserve_strength", type=float, default=None, help="optional SAR->Optical change-preserving strength override")
    parser.add_argument("--sar_cgdr_change_preserve_strength", type=float, default=None, help="optional Optical->SAR change-preserving strength override")
    parser.add_argument("--opt_cgdr_region_mode", type=str, default="correlation", choices=["correlation", "sar_scatter"], help="CGDR region mode for SAR->Optical branch")
    parser.add_argument("--sar_cgdr_region_mode", type=str, default="sar_scatter", choices=["correlation", "sar_scatter"], help="CGDR region mode for Optical->SAR branch")
    parser.add_argument("--cgdr_scatter_threshold", type=float, default=0.55, help="strong-scattering threshold for SAR-domain CGDR")
    parser.add_argument("--lee_filter_sar_cd", type=str2bool, default=True, help="apply Lee filtering before SAR-domain CD")
    parser.add_argument(
        "--sar_lee_filter_mode",
        type=str,
        default="auto",
        choices=["auto", "none", "both", "fixed_only", "moving_only"],
        help="Optical->SAR branch Lee filtering mode; auto keeps legacy bool behavior",
    )
    parser.add_argument(
        "--opt_lee_filter_mode",
        type=str,
        default="none",
        choices=["none", "both", "fixed_only", "moving_only"],
        help="SAR->Optical branch Lee filtering mode (default off)",
    )
    parser.add_argument("--max_test_images", type=int, default=None, help="optional smoke-test limit for processed images")
    parser.add_argument("--save_cgdr_visuals", type=str2bool, default=True, help="save CGDR diagnostic visualizations under result/exp_name/cgdr_visuals")
    parser.add_argument("--max_visual_images", type=int, default=20, help="maximum number of test images with CGDR visualizations")
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
    if opt.ucef_scale < 0.0:
        raise ValueError("ucef_scale must be >= 0.")
    if opt.racr_scale < 0.0:
        raise ValueError("racr_scale must be >= 0.")
    if opt.racr_base_suppress < 0.0:
        raise ValueError("racr_base_suppress must be >= 0.")
    print(opt)

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

    checkpoint_root = "checkpoint"
    prefix = f"{opt.dataset}_HuiYanMTCDN_"
    if not os.path.isdir(checkpoint_root):
        raise FileNotFoundError("checkpoint directory not found.")
    if opt.exp_name is not None and len(opt.exp_name.strip()) > 0:
        exp_name = opt.exp_name.strip()
        if not os.path.isdir(os.path.join(checkpoint_root, exp_name)):
            raise FileNotFoundError(f"Specified exp_name not found: {exp_name}")
    else:
        candidates = [d for d in os.listdir(checkpoint_root) if d.startswith(prefix) and os.path.isdir(os.path.join(checkpoint_root, d))]
        if not candidates:
            raise FileNotFoundError(f"No experiment folder found in checkpoint/ with prefix: {prefix}")
        exp_name = max(candidates, key=lambda d: os.path.getmtime(os.path.join(checkpoint_root, d)))
    print("Using experiment:", exp_name)

    result_dir = os.path.join("result", exp_name)
    optical_translate_dir = os.path.join(result_dir, "optical_tranlation")
    optical_cd_dir = os.path.join(result_dir, "optical_CD_result")
    sar_translate_dir = os.path.join(result_dir, "SAR_tranlation")
    sar_cd_dir = os.path.join(result_dir, "SAR_CD_result")
    optical_confusion_dir = os.path.join(result_dir, "optical_confusion_result")
    sar_confusion_dir = os.path.join(result_dir, "SAR_confusion_result")
    metrics_dir = os.path.join(result_dir, "metrics")
    cgdr_visual_dir = os.path.join(result_dir, "cgdr_visuals")
    generated_sar_dir = os.path.join("results", "generated_sar", exp_name)
    os.makedirs(optical_translate_dir, exist_ok=True)
    os.makedirs(optical_cd_dir, exist_ok=True)
    os.makedirs(sar_translate_dir, exist_ok=True)
    os.makedirs(sar_cd_dir, exist_ok=True)
    os.makedirs(optical_confusion_dir, exist_ok=True)
    os.makedirs(sar_confusion_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    if opt.save_cgdr_visuals:
        os.makedirs(cgdr_visual_dir, exist_ok=True)
    os.makedirs(generated_sar_dir, exist_ok=True)

    model_G_opt2sar = os.path.join("checkpoint", exp_name, "G_opt2sar_best.pth")
    model_G_sar2opt = os.path.join("checkpoint", exp_name, "G_sar2opt_best.pth")
    model_D_opt = os.path.join("checkpoint", exp_name, "D_opt_best.pth")
    model_D_sar = os.path.join("checkpoint", exp_name, "D_sar_best.pth")

    if not (os.path.exists(model_G_opt2sar) and os.path.exists(model_G_sar2opt) and os.path.exists(model_D_opt) and os.path.exists(model_D_sar)):
        raise FileNotFoundError("Best checkpoint files are incomplete. Train first or check checkpoint paths.")

    input_shape = (opt.input_nc, opt.img_height, opt.img_width)
    G_opt2sar = build_generator(opt.input_nc, opt.output_nc, opt.n_residual_blocks)
    G_sar2opt = build_generator(opt.input_nc, opt.output_nc, opt.n_residual_blocks)
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
        cgdr_mask_alignment_with_valid=opt.cgdr_mask_alignment_with_valid,
        cgdr_change_preserve_strength=opt.opt_cgdr_change_preserve_strength,
        cgdr_change_preserve_threshold=opt.cgdr_change_preserve_threshold,
        cgdr_change_preserve_temperature=opt.cgdr_change_preserve_temperature,
        cgdr_use_speckle_filter=opt.cgdr_use_speckle_filter,
        cgdr_use_coarse_fine_split=opt.cgdr_use_coarse_fine_split,
        filter_speckle_for_cd=False,
        speckle_filter_mode=opt.opt_lee_filter_mode,
        use_ucef=opt.use_ucef,
        ucef_scale=opt.ucef_scale,
        use_racr=opt.use_racr,
        racr_scale=opt.racr_scale,
        racr_base_suppress=opt.racr_base_suppress,
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
        cgdr_mask_alignment_with_valid=opt.cgdr_mask_alignment_with_valid,
        cgdr_change_preserve_strength=opt.sar_cgdr_change_preserve_strength,
        cgdr_change_preserve_threshold=opt.cgdr_change_preserve_threshold,
        cgdr_change_preserve_temperature=opt.cgdr_change_preserve_temperature,
        cgdr_use_speckle_filter=opt.cgdr_use_speckle_filter,
        cgdr_use_coarse_fine_split=opt.cgdr_use_coarse_fine_split,
        filter_speckle_for_cd=opt.lee_filter_sar_cd,
        speckle_filter_mode=opt.sar_lee_filter_mode,
        use_ucef=opt.use_ucef,
        ucef_scale=opt.ucef_scale,
        use_racr=opt.use_racr,
        racr_scale=opt.racr_scale,
        racr_base_suppress=opt.racr_base_suppress,
    )

    G_opt2sar.load_state_dict(torch.load(model_G_opt2sar, map_location="cpu"))
    G_sar2opt.load_state_dict(torch.load(model_G_sar2opt, map_location="cpu"))
    missing_opt, unexpected_opt = D_opt.load_state_dict(torch.load(model_D_opt, map_location="cpu"), strict=False)
    missing_sar, unexpected_sar = D_sar.load_state_dict(torch.load(model_D_sar, map_location="cpu"), strict=False)
    if missing_opt or unexpected_opt:
        print(f"[WARN] D_opt checkpoint mismatch: missing={len(missing_opt)}, unexpected={len(unexpected_opt)}")
    if missing_sar or unexpected_sar:
        print(f"[WARN] D_sar checkpoint mismatch: missing={len(missing_sar)}, unexpected={len(unexpected_sar)}")

    G_opt2sar = G_opt2sar.to(device)
    G_sar2opt = G_sar2opt.to(device)
    D_opt = D_opt.to(device)
    D_sar = D_sar.to(device)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        G_opt2sar = nn.DataParallel(G_opt2sar)
        G_sar2opt = nn.DataParallel(G_sar2opt)
        D_opt = nn.DataParallel(D_opt)
        D_sar = nn.DataParallel(D_sar)

    image_dir = os.path.join("dataset", opt.dataset, "test", "Image")
    image_dir2 = os.path.join("dataset", opt.dataset, "test", "Image2")
    label_dir = os.path.join("dataset", opt.dataset, "test", "label")

    image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]
    if opt.max_test_images is not None and opt.max_test_images > 0:
        image_filenames = image_filenames[: opt.max_test_images]
    image2_stem_map = {os.path.splitext(x)[0]: x for x in os.listdir(image_dir2) if is_image_file(x)}
    label_stem_map = {os.path.splitext(x)[0]: x for x in os.listdir(label_dir) if is_image_file(x)}

    A_TP = A_TN = A_FP = A_FN = 0.0
    B_TP = B_TN = B_FP = B_FN = 0.0
    per_image_metrics = []
    skipped_images = []
    visual_count = 0

    G_opt2sar.eval()
    G_sar2opt.eval()
    D_opt.eval()
    D_sar.eval()

    with torch.no_grad():
        for image_name in image_filenames:
            t0 = time.time()
            stem = os.path.splitext(image_name)[0]
            image2_name = image2_stem_map.get(stem)
            label_name = label_stem_map.get(stem)
            if image2_name is None or label_name is None:
                skipped_images.append(image_name)
                continue

            img1 = Image.open(os.path.join(image_dir, image_name)).convert("RGB")
            img1 = np.array(img1) / 255.0
            img1 = np.transpose(img1, (2, 0, 1))

            img2 = Image.open(os.path.join(image_dir2, image2_name)).convert("RGB")
            img2 = np.array(img2) / 255.0
            img2 = np.transpose(img2, (2, 0, 1))

            label = Image.open(os.path.join(label_dir, label_name))
            label = np.array(label)
            lbl = np.where(label > 0, 1, label)
            lbl = torch.tensor(lbl).to(device, dtype=torch.long)

            real_A = to_neg_one_pos_one(torch.tensor(img1).unsqueeze(0).to(device, dtype=torch.float))
            real_B = to_neg_one_pos_one(torch.tensor(img2).unsqueeze(0).to(device, dtype=torch.float))

            fake_A = G_sar2opt(real_B)
            fake_B = G_opt2sar(real_A)
            need_visual = opt.save_cgdr_visuals and (
                opt.max_visual_images is None or opt.max_visual_images < 0 or visual_count < opt.max_visual_images
            )
            if need_visual:
                _, _, output_A, reg_aux_A = D_opt(real_A, fake_A, return_registration_aux=True)
                _, _, output_B, reg_aux_B = D_sar(real_B, fake_B, return_registration_aux=True)
            else:
                _, _, output_A = D_opt(real_A, fake_A)
                _, _, output_B = D_sar(real_B, fake_B)
                reg_aux_A = None
                reg_aux_B = None

            out_img_A = tensor_to_uint8_img(fake_A)
            out_img_B = tensor_to_uint8_img(fake_B)

            pred_A = torch.argmax(output_A, 1).squeeze()
            pred_B = torch.argmax(output_B, 1).squeeze()

            a_tp, a_tn, a_fp, a_fn = confusion_counts(pred_A, lbl)
            b_tp, b_tn, b_fp, b_fn = confusion_counts(pred_B, lbl)
            A_TP += a_tp
            A_TN += a_tn
            A_FP += a_fp
            A_FN += a_fn
            B_TP += b_tp
            B_TN += b_tn
            B_FP += b_fp
            B_FN += b_fn

            a_metrics = metric_dict(a_tp, a_tn, a_fp, a_fn)
            b_metrics = metric_dict(b_tp, b_tn, b_fp, b_fn)
            per_image_metrics.append(
                {
                    "image": image_name,
                    "sar_image": image2_name,
                    "label": label_name,
                    "optical_tp": a_metrics["tp"],
                    "optical_tn": a_metrics["tn"],
                    "optical_fp": a_metrics["fp"],
                    "optical_fn": a_metrics["fn"],
                    "optical_oa": a_metrics["oa"],
                    "optical_precision": a_metrics["precision"],
                    "optical_recall": a_metrics["recall"],
                    "optical_f1": a_metrics["f1"],
                    "optical_iou": a_metrics["iou"],
                    "sar_tp": b_metrics["tp"],
                    "sar_tn": b_metrics["tn"],
                    "sar_fp": b_metrics["fp"],
                    "sar_fn": b_metrics["fn"],
                    "sar_oa": b_metrics["oa"],
                    "sar_precision": b_metrics["precision"],
                    "sar_recall": b_metrics["recall"],
                    "sar_f1": b_metrics["f1"],
                    "sar_iou": b_metrics["iou"],
                }
            )

            a = pred_A.cpu().numpy().astype(np.uint8)
            a = np.where(a > 0, 255, a)
            Image.fromarray(a).save(os.path.join(optical_cd_dir, image_name))

            b = pred_B.cpu().numpy().astype(np.uint8)
            b = np.where(b > 0, 255, b)
            Image.fromarray(b).save(os.path.join(sar_cd_dir, image_name))
            label_binary = lbl.detach().cpu().numpy().astype(np.uint8)
            save_confusion_map(
                pred_A.detach().cpu().numpy().astype(np.uint8),
                label_binary,
                os.path.join(optical_confusion_dir, f"{stem}.png"),
            )
            save_confusion_map(
                pred_B.detach().cpu().numpy().astype(np.uint8),
                label_binary,
                os.path.join(sar_confusion_dir, f"{stem}.png"),
            )

            Image.fromarray(out_img_A).save(os.path.join(optical_translate_dir, image_name))
            Image.fromarray(out_img_B).save(os.path.join(sar_translate_dir, image_name))
            if opt.save_generated_images:
                Image.fromarray(out_img_B).save(os.path.join(generated_sar_dir, image_name))

            if need_visual:
                visual_root = os.path.join(cgdr_visual_dir, stem)
                os.makedirs(visual_root, exist_ok=True)
                Image.fromarray(tensor_to_uint8_img(real_A)).save(os.path.join(visual_root, "real_optical.png"))
                Image.fromarray(tensor_to_uint8_img(real_B)).save(os.path.join(visual_root, "real_sar.png"))
                Image.fromarray(out_img_A).save(os.path.join(visual_root, "generated_optical.png"))
                Image.fromarray(out_img_B).save(os.path.join(visual_root, "generated_sar.png"))
                save_cd_map(lbl.detach().cpu(), os.path.join(visual_root, "label.png"))
                label_img = lbl.detach().cpu().numpy().astype(np.uint8)
                label_img = np.where(label_img > 0, 255, 0).astype(np.uint8)
                pred_A_img = pred_A.detach().cpu().numpy().astype(np.uint8)
                pred_A_img = np.where(pred_A_img > 0, 255, 0).astype(np.uint8)
                pred_B_img = pred_B.detach().cpu().numpy().astype(np.uint8)
                pred_B_img = np.where(pred_B_img > 0, 255, 0).astype(np.uint8)
                save_visual_panel(
                    [
                        ("real optical", tensor_to_uint8_img(real_A)),
                        ("real SAR", tensor_to_uint8_img(real_B)),
                        ("gen optical", out_img_A),
                        ("gen SAR", out_img_B),
                        ("label", label_img),
                        ("CD optical", pred_A_img),
                        ("CD SAR", pred_B_img),
                    ],
                    os.path.join(visual_root, "overview_panel.png"),
                    cols=4,
                )
                save_cgdr_branch_visuals(
                    visual_root,
                    "sar_to_optical",
                    real_A,
                    fake_A,
                    pred_A,
                    reg_aux_A,
                    opt.cgdr_scatter_threshold,
                )
                save_cgdr_branch_visuals(
                    visual_root,
                    "optical_to_sar",
                    real_B,
                    fake_B,
                    pred_B,
                    reg_aux_B,
                    opt.cgdr_scatter_threshold,
                )
                visual_count += 1

            print(f"{image_name}: {time.time() - t0:.4f}s")

    oa_A, p_A, r_A, f1_A, iou_A = compute_metrics(A_TP, A_TN, A_FP, A_FN)
    oa_B, p_B, r_B, f1_B, iou_B = compute_metrics(B_TP, B_TN, B_FP, B_FN)

    print("Optical Metrics")
    print(f"OA: {oa_A:.6f} Precision: {p_A:.6f} Recall: {r_A:.6f} F1: {f1_A:.6f} IoU: {iou_A:.6f}")
    print("SAR Metrics")
    print(f"OA: {oa_B:.6f} Precision: {p_B:.6f} Recall: {r_B:.6f} F1: {f1_B:.6f} IoU: {iou_B:.6f}")

    summary_path = os.path.join(metrics_dir, "summary_metrics.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Dataset: {opt.dataset}\n")
        f.write(f"Processed images: {len(per_image_metrics)}\n")
        f.write(f"Skipped images: {len(skipped_images)}\n")
        if skipped_images:
            f.write("Skipped image names: " + ", ".join(skipped_images) + "\n")
        f.write("\n")
        f.write("Optical CD result (SAR -> Optical, detected in optical domain)\n")
        f.write(f"TP: {A_TP:.0f} TN: {A_TN:.0f} FP: {A_FP:.0f} FN: {A_FN:.0f}\n")
        f.write(f"OA: {oa_A:.6f}\n")
        f.write(f"Precision: {p_A:.6f}\n")
        f.write(f"Recall: {r_A:.6f}\n")
        f.write(f"F1: {f1_A:.6f}\n")
        f.write(f"IoU: {iou_A:.6f}\n\n")
        f.write("SAR CD result (Optical -> SAR, detected in SAR domain)\n")
        f.write(f"TP: {B_TP:.0f} TN: {B_TN:.0f} FP: {B_FP:.0f} FN: {B_FN:.0f}\n")
        f.write(f"OA: {oa_B:.6f}\n")
        f.write(f"Precision: {p_B:.6f}\n")
        f.write(f"Recall: {r_B:.6f}\n")
        f.write(f"F1: {f1_B:.6f}\n")
        f.write(f"IoU: {iou_B:.6f}\n")

    csv_path = os.path.join(metrics_dir, "per_image_metrics.csv")
    if per_image_metrics:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_image_metrics[0].keys()))
            writer.writeheader()
            writer.writerows(per_image_metrics)
    print(f"Saved metrics summary: {summary_path}")
    print(f"Saved per-image metrics: {csv_path}")
