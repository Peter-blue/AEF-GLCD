from __future__ import print_function

import argparse
import csv
import datetime as dt
import itertools
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from data import get_test_set, get_training_set
from loss.focalloss import FocalLoss
from Model.Sun_Net_gan import Discriminator, weights_init_normal
from Model.utils import LambdaLR, ReplayBuffer
from models.generator import build_generator, load_pretrained_generator


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y", "t")


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def leaf_modules_with_params(model):
    modules = []
    for m in model.modules():
        if len(list(m.children())) == 0 and len(list(m.parameters(recurse=False))) > 0:
            modules.append(m)
    return modules


def set_all_requires_grad(model, flag):
    for p in model.parameters():
        p.requires_grad = flag


def set_generator_train_stage(model, stage, freeze_first_n=10):
    net = unwrap(model)
    leaf_modules = leaf_modules_with_params(net)

    if stage == "all_frozen":
        set_all_requires_grad(net, False)
    elif stage == "last5":
        set_all_requires_grad(net, False)
        start = min(max(0, int(freeze_first_n)), len(leaf_modules))
        candidates = leaf_modules[start:] if start < len(leaf_modules) else leaf_modules
        train_modules = candidates[-5:] if len(candidates) >= 5 else leaf_modules[-5:]
        for m in train_modules:
            for p in m.parameters(recurse=False):
                p.requires_grad = True
    elif stage == "all_trainable":
        set_all_requires_grad(net, True)
    else:
        raise ValueError(f"Unknown generator stage: {stage}")

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    return trainable, total


def compute_metrics(tp, tn, fp, fn, eps=1e-8):
    oa = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return oa, precision, recall, f1, iou


def compute_selection_score(opt, f1_opt, f1_sar, iou_opt, iou_sar):
    w_opt = float(opt.model_select_opt_weight)
    w_sar = float(opt.model_select_sar_weight)
    norm = max(w_opt + w_sar, 1e-8)
    w_opt /= norm
    w_sar /= norm

    if opt.model_select_metric == "opt_f1":
        return f1_opt
    if opt.model_select_metric == "sar_f1":
        return f1_sar
    if opt.model_select_metric == "mean_iou":
        return w_opt * iou_opt + w_sar * iou_sar
    if opt.model_select_metric == "mean_f1_iou":
        return 0.5 * (w_opt * f1_opt + w_sar * f1_sar) + 0.5 * (w_opt * iou_opt + w_sar * iou_sar)
    # default: mean_f1
    return w_opt * f1_opt + w_sar * f1_sar


def _masked_mean(x, mask=None, eps=1e-8):
    if mask is None:
        return x.mean()
    mask = mask.float()
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return (x * mask).sum() / (mask.sum() + eps)


def _normalized_entropy(prob, eps=1e-8):
    entropy = -(prob * torch.log(prob + eps)).sum(dim=1, keepdim=True)
    return entropy / np.log(max(2, prob.shape[1]))


# Legacy GBCC implementation kept for reproducibility/ablation.
# Current default training path uses SABR branch weighting (see training loop).
def compute_gbcc_loss(
    logits_opt,
    logits_sar,
    loss_cd_opt,
    loss_cd_sar,
    target=None,
    delta=0.20,
    tau=0.20,
    reliability_margin=0.10,
    entropy_weight=0.70,
    loss_weight=0.30,
    use_label_gate=True,
    teacher_unchanged_threshold=0.30,
    teacher_changed_threshold=0.70,
    valid_mask=None,
    eps=1e-8,
):
    prob_opt = torch.softmax(logits_opt, dim=1)
    prob_sar = torch.softmax(logits_sar, dim=1)
    p_opt = prob_opt[:, 1:2]
    p_sar = prob_sar[:, 1:2]

    if valid_mask is None:
        base_mask = torch.ones_like(p_opt)
    else:
        base_mask = valid_mask.float()
        if base_mask.dim() == 3:
            base_mask = base_mask.unsqueeze(1)

    label = None
    unchanged = None
    changed = None
    if target is not None:
        label = target.float()
        if label.dim() == 3:
            label = label.unsqueeze(1)
        unchanged = (label < 0.5).float()
        changed = (label >= 0.5).float()

    entropy_opt = _normalized_entropy(prob_opt, eps=eps).detach()
    entropy_sar = _normalized_entropy(prob_sar, eps=eps).detach()
    conf_opt_map = (1.0 - entropy_opt).clamp(0.0, 1.0)
    conf_sar_map = (1.0 - entropy_sar).clamp(0.0, 1.0)
    score_opt = torch.exp(-loss_cd_opt.detach()).clamp(0.0, 1.0)
    score_sar = torch.exp(-loss_cd_sar.detach()).clamp(0.0, 1.0)

    norm = max(float(entropy_weight + loss_weight), eps)
    r_opt_map = (entropy_weight * conf_opt_map + loss_weight * score_opt) / norm
    r_sar_map = (entropy_weight * conf_sar_map + loss_weight * score_sar) / norm
    reliability_gap = r_opt_map - r_sar_map
    dominance = torch.clamp(
        (torch.abs(reliability_gap) - reliability_margin) / max(1.0 - reliability_margin, eps),
        min=0.0,
        max=1.0,
    )
    reliability_gate = torch.max(r_opt_map, r_sar_map).clamp(0.0, 1.0)
    game_gate = dominance * reliability_gate

    if use_label_gate and label is not None:
        opt_teacher_gate = (
            unchanged * (p_opt.detach() <= teacher_unchanged_threshold).float()
            + changed * (p_opt.detach() >= teacher_changed_threshold).float()
        )
        sar_teacher_gate = (
            unchanged * (p_sar.detach() <= teacher_unchanged_threshold).float()
            + changed * (p_sar.detach() >= teacher_changed_threshold).float()
        )
    else:
        opt_teacher_gate = torch.ones_like(p_opt)
        sar_teacher_gate = torch.ones_like(p_sar)

    w_opt_to_sar = torch.sigmoid(reliability_gap / max(tau, eps)) * game_gate * opt_teacher_gate
    w_sar_to_opt = torch.sigmoid(-reliability_gap / max(tau, eps)) * game_gate * sar_teacher_gate

    if label is not None:
        opt_to_sar_map = (
            unchanged * torch.relu(p_sar - p_opt.detach() - delta).pow(2)
            + changed * torch.relu(p_opt.detach() - p_sar - delta).pow(2)
        )
        sar_to_opt_map = (
            unchanged * torch.relu(p_opt - p_sar.detach() - delta).pow(2)
            + changed * torch.relu(p_sar.detach() - p_opt - delta).pow(2)
        )
    else:
        opt_to_sar_map = torch.relu(torch.abs(p_sar - p_opt.detach()) - delta).pow(2)
        sar_to_opt_map = torch.relu(torch.abs(p_opt - p_sar.detach()) - delta).pow(2)

    denom = base_mask.sum() + eps
    loss_opt_to_sar = (opt_to_sar_map * w_opt_to_sar * base_mask).sum() / denom
    loss_sar_to_opt = (sar_to_opt_map * w_sar_to_opt * base_mask).sum() / denom
    loss = loss_opt_to_sar + loss_sar_to_opt
    aux = {
        "r_opt": float(_masked_mean(r_opt_map.detach(), base_mask, eps).detach().cpu()),
        "r_sar": float(_masked_mean(r_sar_map.detach(), base_mask, eps).detach().cpu()),
        "w_opt_to_sar": float(_masked_mean(w_opt_to_sar.detach(), base_mask, eps).detach().cpu()),
        "w_sar_to_opt": float(_masked_mean(w_sar_to_opt.detach(), base_mask, eps).detach().cpu()),
        "game_gate": float(_masked_mean(game_gate.detach(), base_mask, eps).detach().cpu()),
    }
    return loss, aux


def compute_sabr_weights(
        
    logits_opt,
    logits_sar,
    loss_cd_opt,
    loss_cd_sar,
    loss_reg_opt=None,
    loss_reg_sar=None,
    entropy_weight=0.55,
    cd_weight=0.30,
    reg_weight=0.15,
    temperature=0.50,
    min_weight=0.35,
    max_weight=0.65,
    valid_mask=None,
    eps=1e-8,
):
    prob_opt = torch.softmax(logits_opt, dim=1)
    prob_sar = torch.softmax(logits_sar, dim=1)
    entropy_opt = _normalized_entropy(prob_opt, eps=eps)
    entropy_sar = _normalized_entropy(prob_sar, eps=eps)
    conf_opt = (1.0 - _masked_mean(entropy_opt, valid_mask, eps)).clamp(0.0, 1.0)
    conf_sar = (1.0 - _masked_mean(entropy_sar, valid_mask, eps)).clamp(0.0, 1.0)
    score_cd_opt = torch.exp(-loss_cd_opt.detach()).clamp(0.0, 1.0)
    score_cd_sar = torch.exp(-loss_cd_sar.detach()).clamp(0.0, 1.0)
    score_reg_opt = torch.exp(-loss_reg_opt.detach()).clamp(0.0, 1.0) if loss_reg_opt is not None else conf_opt
    score_reg_sar = torch.exp(-loss_reg_sar.detach()).clamp(0.0, 1.0) if loss_reg_sar is not None else conf_sar

    norm = max(float(entropy_weight + cd_weight + reg_weight), eps)
    r_opt = (entropy_weight * conf_opt + cd_weight * score_cd_opt + reg_weight * score_reg_opt) / norm
    r_sar = (entropy_weight * conf_sar + cd_weight * score_cd_sar + reg_weight * score_reg_sar) / norm
    raw = torch.stack([r_opt, r_sar], dim=0)
    w = torch.softmax(raw / max(float(temperature), eps), dim=0)
    w_opt = torch.clamp(w[0], min=float(min_weight), max=float(max_weight))
    w_sar = torch.clamp(w[1], min=float(min_weight), max=float(max_weight))
    w_sum = (w_opt + w_sar).clamp_min(eps)
    w_opt = w_opt / w_sum
    w_sar = w_sar / w_sum
    aux = {
        "r_opt": float(r_opt.detach().cpu()),
        "r_sar": float(r_sar.detach().cpu()),
        "w_opt": float(w_opt.detach().cpu()),
        "w_sar": float(w_sar.detach().cpu()),
    }
    return w_opt, w_sar, aux


def compute_racr_fp_loss(logits, target, cd_aux=None, gamma=2.0, eps=1e-8):
    if cd_aux is None or "unreliable" not in cd_aux:
        return logits.new_zeros(())
    unreliable = cd_aux["unreliable"].detach()
    if unreliable.shape[-2:] != logits.shape[-2:]:
        unreliable = F.interpolate(unreliable, size=logits.shape[-2:], mode="bilinear", align_corners=False)
    unchanged = (target == 0).float().unsqueeze(1)
    p_change = torch.softmax(logits, dim=1)[:, 1:2]
    weight = unreliable.clamp(0.0, 1.0) * unchanged
    return (weight * p_change.clamp(0.0, 1.0).pow(float(gamma))).sum() / (weight.sum() + eps)


def _target_4d(target):
    target_f = target.float()
    if target_f.dim() == 3:
        target_f = target_f.unsqueeze(1)
    return target_f


def compute_boundary_mask(target, boundary_width=3):
    target_f = _target_4d(target)
    width = max(1, int(boundary_width))
    kernel = 2 * width + 1
    dilated = F.max_pool2d(target_f, kernel_size=kernel, stride=1, padding=width)
    eroded = 1.0 - F.max_pool2d(1.0 - target_f, kernel_size=kernel, stride=1, padding=width)
    return (dilated - eroded).clamp(0.0, 1.0)


def compute_barc_losses(logits, target, boundary_width=3, eps=1e-8):
    target_f = _target_4d(target)
    boundary = compute_boundary_mask(target, boundary_width=boundary_width)

    ce_map = F.cross_entropy(logits, target.long(), reduction="none").unsqueeze(1)
    boundary_loss = (ce_map * boundary).sum() / (boundary.sum() + eps)

    p_change = torch.softmax(logits, dim=1)[:, 1:2]
    dx = torch.abs(p_change[:, :, :, 1:] - p_change[:, :, :, :-1])
    dy = torch.abs(p_change[:, :, 1:, :] - p_change[:, :, :-1, :])
    same_x = (target_f[:, :, :, 1:] == target_f[:, :, :, :-1]).float()
    same_y = (target_f[:, :, 1:, :] == target_f[:, :, :-1, :]).float()
    region_loss_x = (dx * same_x).sum() / (same_x.sum() + eps)
    region_loss_y = (dy * same_y).sum() / (same_y.sum() + eps)
    region_loss = 0.5 * (region_loss_x + region_loss_y)

    return boundary_loss, region_loss


def to_neg_one_pos_one(x: torch.Tensor) -> torch.Tensor:
    # Dataset gives [0,1], HuiYan generator is trained around [-1,1]
    return x * 2.0 - 1.0


def to_zero_one(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1.0, 1.0) + 1.0) * 0.5


def resolve_optional(value, fallback):
    return fallback if value is None else value


def resolve_speckle_mode(legacy_flag, mode, default_when_true="both"):
    if mode is None or mode == "auto":
        return default_when_true if legacy_flag else "none"
    return mode


def build_loader(dataset, batch_size, shuffle, drop_last, opt):
    loader_kwargs = {
        "num_workers": opt.num_workers,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "pin_memory": opt.pin_memory and torch.cuda.is_available(),
    }
    if opt.num_workers > 0:
        loader_kwargs["persistent_workers"] = opt.persistent_workers
        loader_kwargs["prefetch_factor"] = opt.prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def count_parameters(model):
    net = unwrap(model)
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total, trainable


def _first_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (list, tuple)):
        for item in x:
            t = _first_tensor(item)
            if t is not None:
                return t
    if isinstance(x, dict):
        for item in x.values():
            t = _first_tensor(item)
            if t is not None:
                return t
    return None


def estimate_model_macs(model, model_inputs):
    net = unwrap(model)
    counters = {"macs": 0.0}
    handles = []

    def add_macs(value):
        counters["macs"] += float(value)

    def conv2d_hook(m, inp, out):
        out_t = _first_tensor(out)
        if out_t is None or out_t.ndim < 4:
            return
        out_elements = out_t.shape[0] * out_t.shape[1] * out_t.shape[2] * out_t.shape[3]
        kernel_mul = (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        add_macs(out_elements * kernel_mul)
        if m.bias is not None:
            add_macs(out_elements)

    def conv_transpose2d_hook(m, inp, out):
        out_t = _first_tensor(out)
        if out_t is None or out_t.ndim < 4:
            return
        out_elements = out_t.shape[0] * out_t.shape[1] * out_t.shape[2] * out_t.shape[3]
        kernel_mul = (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        add_macs(out_elements * kernel_mul)
        if m.bias is not None:
            add_macs(out_elements)

    def linear_hook(m, inp, out):
        in_t = _first_tensor(inp)
        out_t = _first_tensor(out)
        if in_t is None or out_t is None:
            return
        out_items = int(np.prod(out_t.shape[:-1])) if out_t.ndim > 1 else 1
        add_macs(out_items * m.in_features * m.out_features)
        if m.bias is not None:
            add_macs(out_t.numel())

    def batchnorm2d_hook(m, inp, out):
        out_t = _first_tensor(out)
        if out_t is not None:
            add_macs(out_t.numel() * 2)

    def activation_hook(m, inp, out):
        out_t = _first_tensor(out)
        if out_t is not None:
            add_macs(out_t.numel())

    def pool2d_hook(m, inp, out):
        out_t = _first_tensor(out)
        if out_t is None or out_t.ndim < 4:
            return
        if isinstance(m.kernel_size, tuple):
            k_h, k_w = m.kernel_size
        else:
            k_h = k_w = m.kernel_size
        add_macs(out_t.numel() * k_h * k_w)

    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv2d_hook))
        elif isinstance(module, nn.ConvTranspose2d):
            handles.append(module.register_forward_hook(conv_transpose2d_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, nn.BatchNorm2d):
            handles.append(module.register_forward_hook(batchnorm2d_hook))
        elif isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)):
            handles.append(module.register_forward_hook(activation_hook))
        elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
            handles.append(module.register_forward_hook(pool2d_hook))

    was_training = net.training
    net.eval()
    with torch.no_grad():
        net(*model_inputs)
    if was_training:
        net.train()
    for h in handles:
        h.remove()
    return counters["macs"]


def format_count(value):
    value = float(value)
    if value >= 1e12:
        return f"{value / 1e12:.4f}T"
    if value >= 1e9:
        return f"{value / 1e9:.4f}G"
    if value >= 1e6:
        return f"{value / 1e6:.4f}M"
    if value >= 1e3:
        return f"{value / 1e3:.4f}K"
    return f"{value:.0f}"


def save_model_profile(rows, profile_txt_path, profile_csv_path, profile_note):
    if not rows:
        return
    with open(profile_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "params_total",
                "params_trainable",
                "macs",
                "flops",
                "params_total_fmt",
                "params_trainable_fmt",
                "macs_fmt",
                "flops_fmt",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    total_params = sum(float(r["params_total"]) for r in rows)
    total_trainable = sum(float(r["params_trainable"]) for r in rows)
    total_macs = sum(float(r["macs"]) for r in rows)
    total_flops = sum(float(r["flops"]) for r in rows)

    with open(profile_txt_path, "w", encoding="utf-8") as f:
        f.write("Model Complexity Profile\n")
        f.write(profile_note + "\n\n")
        for r in rows:
            f.write(
                f"{r['model']}: params={r['params_total_fmt']} (trainable={r['params_trainable_fmt']}), "
                f"MACs={r['macs_fmt']}, FLOPs={r['flops_fmt']}\n"
            )
        f.write("\n")
        f.write(
            f"TOTAL: params={format_count(total_params)} (trainable={format_count(total_trainable)}), "
            f"MACs={format_count(total_macs)}, FLOPs={format_count(total_flops)}\n"
        )


def maybe_plot_epoch_curves(epoch_rows, out_path):
    if not epoch_rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Skip curve plotting because matplotlib is unavailable: {exc}")
        return

    epochs = [int(r["epoch"]) for r in epoch_rows]
    opt_f1 = [float(r["opt_f1"]) for r in epoch_rows]
    sar_f1 = [float(r["sar_f1"]) for r in epoch_rows]
    opt_iou = [float(r["opt_iou"]) for r in epoch_rows]
    sar_iou = [float(r["sar_iou"]) for r in epoch_rows]
    epoch_time = [float(r["epoch_time_s"]) for r in epoch_rows]
    train_time = [float(r["train_time_s"]) for r in epoch_rows]
    eval_time = [float(r["eval_time_s"]) for r in epoch_rows]
    g_loss = [float(r["train_g_loss"]) for r in epoch_rows]
    d_loss = [float(r["train_d_loss"]) for r in epoch_rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(epochs, opt_f1, label="Optical F1", linewidth=2)
    axes[0, 0].plot(epochs, sar_f1, label="SAR F1", linewidth=2)
    axes[0, 0].set_title("F1 Curve")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("F1")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, opt_iou, label="Optical IoU", linewidth=2)
    axes[0, 1].plot(epochs, sar_iou, label="SAR IoU", linewidth=2)
    axes[0, 1].set_title("IoU Curve")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("IoU")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, epoch_time, label="Epoch Total Time", linewidth=2)
    axes[1, 0].plot(epochs, train_time, label="Train Time", linestyle="--")
    axes[1, 0].plot(epochs, eval_time, label="Eval Time", linestyle="--")
    axes[1, 0].set_title("Time Curve")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Seconds")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, g_loss, label="Train G Loss", linewidth=2)
    axes[1, 1].plot(epochs, d_loss, label="Train D Loss", linewidth=2)
    axes[1, 1].set_title("Loss Curve")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MTCDN + HuiYan generator end-to-end training")
    parser.add_argument("--dataset", required=False, default="Gloucester")
    parser.add_argument("--batch_size", type=int, default=2, help="training batch size")
    parser.add_argument("--test_batch_size", type=int, default=1, help="testing batch size")
    parser.add_argument("--direction", type=str, default="a2b", help="a2b or b2a")
    parser.add_argument("--input_nc", type=int, default=3, help="input image channels")
    parser.add_argument("--output_nc", type=int, default=3, help="output image channels")
    parser.add_argument("--ngf", type=int, default=64, help="generator filters in first conv layer")
    parser.add_argument("--ndf", type=int, default=512, help="discriminator filters in first conv layer")
    parser.add_argument("--epoch_count", type=int, default=0, help="the starting epoch count")
    parser.add_argument("--niter", type=int, default=500, help="# of epochs at starting learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="alias of niter")
    parser.add_argument("--niter_decay", type=int, default=100, help="# of epochs to linearly decay learning rate to zero")
    parser.add_argument("--epoch", type=int, default=0, help="resume flag: 0 means train from scratch")
    parser.add_argument("--run_name", type=str, default=None, help="optional suffix for experiment name; default uses timestamp")
    parser.add_argument("--G_lr", type=float, default=1e-5, help="generator learning rate")
    parser.add_argument("--D_lr", type=float, default=1e-4, help="discriminator learning rate")
    parser.add_argument("--lr_policy", type=str, default="lambda", help="learning rate policy")
    parser.add_argument("--lr_decay_iters", type=int, default=100, help="multiply by a gamma every lr_decay_iters iterations")
    parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use when CUDA is available")
    parser.add_argument("--threads", type=int, default=4, help="number of threads for data loader")
    parser.add_argument("--num_workers", type=int, default=0, help="number of DataLoader worker processes")
    parser.add_argument("--pin_memory", type=str2bool, default=True, help="enable pinned memory for faster CPU-to-GPU transfer")
    parser.add_argument("--persistent_workers", type=str2bool, default=True, help="keep DataLoader workers alive across epochs")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="number of batches prefetched by each DataLoader worker")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--lamb", type=int, default=10, help="unused legacy weight")
    parser.add_argument("--img_height", type=int, default=256, help="image height")
    parser.add_argument("--img_width", type=int, default=256, help="image width")
    parser.add_argument("--b1", type=float, default=0.5, help="adam beta1")
    parser.add_argument("--b2", type=float, default=0.999, help="adam beta2")
    parser.add_argument("--sample_interval", type=int, default=5, help="interval between saving generator outputs")
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between periodic checkpoints; -1 keeps best only")
    parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
    parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight (lambda2)")
    parser.add_argument("--lambda_id", type=float, default=5.0, help="recon/identity loss weight (lambda1)")
    parser.add_argument("--freeze_generator_layers", type=int, default=10, help="number of front layers to freeze in partial fine-tune")
    parser.add_argument("--save_generated_images", type=str2bool, default=False, help="whether to save generated samples during training")
    parser.add_argument("--freeze_stage0_epochs", type=int, default=5, help="epochs for fully freezing generator when pretrained is valid")
    parser.add_argument("--partial_finetune_until", type=int, default=80, help="epoch boundary for partial generator fine-tuning")
    parser.add_argument("--use_pretrained", type=str2bool, default=True, help="whether to load HuiYan pretrained generator weights")
    parser.add_argument("--use_cgdr", type=str2bool, default=True, help="enable correlation-guided deformable registration")
    parser.add_argument("--use_ucef", type=str2bool, default=False, help="enable uncertainty-aware change evidence fusion in SNUNet_ECAM")
    parser.add_argument("--ucef_scale", type=float, default=0.5, help="residual scale of UCEF evidence enhancement")
    parser.add_argument("--use_racr", type=str2bool, default=False, help="enable registration-aware change refinement in SNUNet_ECAM")
    parser.add_argument("--racr_scale", type=float, default=0.2, help="learnable logit-refinement scale for RACR")
    parser.add_argument("--racr_base_suppress", type=float, default=0.05, help="conservative base suppression for unreliable change responses")
    parser.add_argument("--lambda_racr_fp", type=float, default=0.02, help="false-positive suppression loss weight for RACR")
    parser.add_argument("--racr_start_epoch", type=int, default=30, help="epoch to start RACR false-positive loss")
    parser.add_argument("--racr_warmup_epochs", type=int, default=50, help="warmup epochs for RACR false-positive loss")
    parser.add_argument("--racr_fp_gamma", type=float, default=2.0, help="focusing exponent for RACR false-positive loss")
    parser.add_argument("--use_barc", type=str2bool, default=False, help="enable boundary-aware region consistency loss for CD")
    parser.add_argument("--lambda_barc_boundary", type=float, default=0.02, help="boundary-aware CD loss weight")
    parser.add_argument("--lambda_barc_region", type=float, default=0.01, help="intra-region consistency loss weight")
    parser.add_argument("--barc_start_epoch", type=int, default=20, help="epoch to start BARC losses")
    parser.add_argument("--barc_warmup_epochs", type=int, default=50, help="warmup epochs for BARC losses")
    parser.add_argument("--barc_boundary_width", type=int, default=3, help="label boundary band width for BARC")
    parser.add_argument("--cgdr_max_flow", type=float, default=6.0, help="maximum coarse CGDR displacement in pixels")
    parser.add_argument("--cgdr_corr_threshold", type=float, default=0.45, help="hard fine-alignment threshold for CGDR")
    parser.add_argument("--lambda_cgdr", type=float, default=0.02, help="weight for CGDR consistency regularization")
    parser.add_argument("--cgdr_warmup_epochs", type=int, default=20, help="linearly warm up CGDR loss weight")
    parser.add_argument("--cgdr_adaptive_gate_alpha", type=float, default=0.25, help="blend ratio for adaptive gate threshold")
    parser.add_argument("--cgdr_target_high_ratio", type=float, default=0.30, help="target ratio for high-confidence fine alignment")
    parser.add_argument("--cgdr_min_high_ratio", type=float, default=0.10, help="minimum allowed high-confidence ratio")
    parser.add_argument("--cgdr_max_high_ratio", type=float, default=0.60, help="maximum allowed high-confidence ratio")
    parser.add_argument("--cgdr_residual_suppress", type=float, default=0.35, help="suppress fine alignment in high-residual regions")
    parser.add_argument("--cgdr_low_conf_flow_scale", type=float, default=0.55, help="scale coarse flow in low-confidence regions")
    parser.add_argument("--cgdr_residual_conf_temperature", type=float, default=6.0, help="residual-to-confidence temperature")
    parser.add_argument("--cgdr_mask_alignment_with_valid", type=str2bool, default=True, help="apply registration only on valid unchanged mask during training")
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
    parser.add_argument("--lambda_cgdr_opt", type=float, default=None, help="optional CGDR regularization weight for SAR->Optical branch")
    parser.add_argument("--lambda_cgdr_sar", type=float, default=None, help="optional CGDR regularization weight for Optical->SAR branch")
    parser.add_argument("--cgdr_warmup_epochs_opt", type=int, default=None, help="optional CGDR warmup epochs for SAR->Optical branch")
    parser.add_argument("--cgdr_warmup_epochs_sar", type=int, default=None, help="optional CGDR warmup epochs for Optical->SAR branch")
    parser.add_argument("--use_sabr", type=str2bool, default=False, help="enable optional scene-adaptive branch reliability regulation")
    parser.add_argument("--sabr_start_epoch", type=int, default=40, help="epoch to start SABR")
    parser.add_argument("--sabr_warmup_epochs", type=int, default=80, help="warmup epochs after SABR starts")
    parser.add_argument("--sabr_entropy_weight", type=float, default=0.55, help="entropy-confidence weight in SABR")
    parser.add_argument("--sabr_cd_weight", type=float, default=0.30, help="CD supervision weight in SABR")
    parser.add_argument("--sabr_reg_weight", type=float, default=0.15, help="registration-consistency weight in SABR")
    parser.add_argument("--sabr_temperature", type=float, default=0.50, help="softmax temperature for SABR branch weights")
    parser.add_argument("--sabr_min_weight", type=float, default=0.35, help="minimum branch weight in SABR")
    parser.add_argument("--sabr_max_weight", type=float, default=0.65, help="maximum branch weight in SABR")
    parser.add_argument("--sabr_use_unchanged_mask", type=str2bool, default=True, help="apply unchanged-area mask when estimating SABR reliability")
    parser.add_argument("--use_gbcc", type=str2bool, default=False, help="enable game-based bidirectional CD constraint (legacy; SABR is recommended)")
    parser.add_argument("--lambda_gbcc", type=float, default=0.002, help="weight for GBCC")
    parser.add_argument("--gbcc_start_epoch", type=int, default=60, help="epoch to start GBCC")
    parser.add_argument("--gbcc_warmup_epochs", type=int, default=100, help="warmup epochs after GBCC starts")
    parser.add_argument("--gbcc_delta", type=float, default=0.20, help="free consistency interval before GBCC penalty")
    parser.add_argument("--gbcc_reliability_tau", type=float, default=0.20, help="temperature for GBCC branch game weights")
    parser.add_argument("--gbcc_reliability_margin", type=float, default=0.10, help="minimum reliability gap before enabling GBCC")
    parser.add_argument("--gbcc_entropy_weight", type=float, default=0.70, help="entropy-confidence weight in GBCC reliability")
    parser.add_argument("--gbcc_loss_weight", type=float, default=0.30, help="supervised-loss weight in GBCC reliability")
    parser.add_argument("--gbcc_use_label_gate", type=str2bool, default=True, help="only trust teacher predictions that agree with labels")
    parser.add_argument("--gbcc_teacher_unchanged_threshold", type=float, default=0.30, help="teacher max change probability in unchanged regions")
    parser.add_argument("--gbcc_teacher_changed_threshold", type=float, default=0.70, help="teacher min change probability in changed regions")
    parser.add_argument("--gbcc_use_unchanged_mask", type=str2bool, default=True, help="apply unchanged-area mask when computing GBCC")
    parser.add_argument("--lambda_cdcc", type=float, default=None, help="legacy alias for lambda_gbcc")
    parser.add_argument("--cdcc_warmup_epochs", type=int, default=None, help="legacy alias for gbcc_warmup_epochs")
    parser.add_argument("--cdcc_loss_type", type=str, default="mse", choices=["mse", "js"], help="legacy compatibility arg")
    parser.add_argument("--cdcc_use_unchanged_mask", type=str2bool, default=None, help="legacy alias for gbcc_use_unchanged_mask")
    parser.add_argument("--opt_cgdr_region_mode", type=str, default="correlation", choices=["correlation", "sar_scatter"], help="CGDR region mode for SAR->Optical branch")
    parser.add_argument("--sar_cgdr_region_mode", type=str, default="sar_scatter", choices=["correlation", "sar_scatter"], help="CGDR region mode for Optical->SAR branch")
    parser.add_argument("--sar_scatter_start_epoch", type=int, default=-1, help="if >=0, switch Optical->SAR branch from correlation to sar_scatter at this epoch")
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
    parser.add_argument("--opt_reg_use_unchanged_mask", type=str2bool, default=True, help="apply unchanged-area mask to SAR->Optical CGDR regularization")
    parser.add_argument("--sar_reg_use_unchanged_mask", type=str2bool, default=True, help="apply unchanged-area mask to Optical->SAR CGDR regularization")
    parser.add_argument(
        "--model_select_metric",
        type=str,
        default="mean_f1",
        choices=["mean_f1", "mean_iou", "mean_f1_iou", "opt_f1", "sar_f1"],
        help="metric used for best-model saving and early stopping",
    )
    parser.add_argument("--model_select_opt_weight", type=float, default=0.5, help="optical branch weight for combined selection metrics")
    parser.add_argument("--model_select_sar_weight", type=float, default=0.5, help="SAR branch weight for combined selection metrics")
    parser.add_argument("--early_stop_patience", type=int, default=50, help="early-stop patience on selection metric")
    parser.add_argument("--early_stop_min_epoch", type=int, default=30, help="minimum epoch before enabling early stop")
    parser.add_argument("--export_epoch_metrics", type=str2bool, default=False, help="save epoch-level metrics/time csv for curve analysis")
    parser.add_argument("--plot_epoch_curves", type=str2bool, default=False, help="save epoch curve figure (.png) after training")
    parser.add_argument("--profile_complexity", type=str2bool, default=False, help="estimate Params/MACs/FLOPs and save to metrics/")
    parser.add_argument("--profile_input_batch", type=int, default=1, help="batch size used for Params/FLOPs profile input")
    parser.add_argument("--profile_only", type=str2bool, default=False, help="only run Params/FLOPs profile then exit")
    parser.add_argument(
        "--pretrained_opt2sar_path",
        type=str,
        default=os.path.join("HuiYanEarth-SAR", "pretrained", "huiyan_sar_v1.pth"),
        help="optical->SAR pretrained checkpoint path",
    )
    parser.add_argument(
        "--pretrained_sar2opt_path",
        type=str,
        default=os.path.join("HuiYanEarth-SAR", "pretrained", "huiyan_opt_v1.pth"),
        help="SAR->optical pretrained checkpoint path",
    )
    opt = parser.parse_args()
    if opt.epochs is not None:
        opt.niter = opt.epochs
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
    opt.lambda_cgdr_opt = float(resolve_optional(opt.lambda_cgdr_opt, opt.lambda_cgdr))
    opt.lambda_cgdr_sar = float(resolve_optional(opt.lambda_cgdr_sar, opt.lambda_cgdr))
    opt.cgdr_warmup_epochs_opt = int(resolve_optional(opt.cgdr_warmup_epochs_opt, opt.cgdr_warmup_epochs))
    opt.cgdr_warmup_epochs_sar = int(resolve_optional(opt.cgdr_warmup_epochs_sar, opt.cgdr_warmup_epochs))
    opt.sar_lee_filter_mode = resolve_speckle_mode(opt.lee_filter_sar_cd, opt.sar_lee_filter_mode, default_when_true="both")
    if opt.lambda_cdcc is not None:
        opt.lambda_gbcc = float(opt.lambda_cdcc)
    if opt.cdcc_warmup_epochs is not None:
        opt.gbcc_warmup_epochs = int(opt.cdcc_warmup_epochs)
    if opt.cdcc_use_unchanged_mask is not None:
        opt.gbcc_use_unchanged_mask = bool(opt.cdcc_use_unchanged_mask)
    if opt.sabr_start_epoch < 0:
        raise ValueError("sabr_start_epoch must be >= 0.")
    if opt.sabr_warmup_epochs < 1:
        raise ValueError("sabr_warmup_epochs must be >= 1.")
    if opt.sabr_temperature <= 0.0:
        raise ValueError("sabr_temperature must be > 0.")
    if opt.sabr_min_weight < 0.0 or opt.sabr_max_weight > 1.0:
        raise ValueError("sabr_min_weight and sabr_max_weight must be in [0, 1].")
    if opt.sabr_min_weight >= opt.sabr_max_weight:
        raise ValueError("sabr_min_weight must be smaller than sabr_max_weight.")
    if opt.sabr_entropy_weight < 0.0 or opt.sabr_cd_weight < 0.0 or opt.sabr_reg_weight < 0.0:
        raise ValueError("sabr_entropy_weight/sabr_cd_weight/sabr_reg_weight must be >= 0.")
    if opt.sabr_entropy_weight + opt.sabr_cd_weight + opt.sabr_reg_weight <= 0.0:
        raise ValueError("sabr weight sum must be > 0.")
    if opt.lambda_gbcc < 0.0:
        raise ValueError("lambda_gbcc must be >= 0.")
    if opt.gbcc_start_epoch < 0:
        raise ValueError("gbcc_start_epoch must be >= 0.")
    if opt.gbcc_warmup_epochs < 1:
        raise ValueError("gbcc_warmup_epochs must be >= 1.")
    if opt.gbcc_delta < 0.0:
        raise ValueError("gbcc_delta must be >= 0.")
    if opt.gbcc_reliability_tau <= 0.0:
        raise ValueError("gbcc_reliability_tau must be > 0.")
    if opt.gbcc_reliability_margin < 0.0 or opt.gbcc_reliability_margin >= 1.0:
        raise ValueError("gbcc_reliability_margin must be in [0, 1).")
    if opt.gbcc_entropy_weight < 0.0 or opt.gbcc_loss_weight < 0.0:
        raise ValueError("gbcc_entropy_weight and gbcc_loss_weight must be >= 0.")
    if opt.gbcc_entropy_weight + opt.gbcc_loss_weight <= 0.0:
        raise ValueError("gbcc_entropy_weight + gbcc_loss_weight must be > 0.")
    if not (0.0 <= opt.gbcc_teacher_unchanged_threshold <= 1.0):
        raise ValueError("gbcc_teacher_unchanged_threshold must be in [0, 1].")
    if not (0.0 <= opt.gbcc_teacher_changed_threshold <= 1.0):
        raise ValueError("gbcc_teacher_changed_threshold must be in [0, 1].")
    if opt.gbcc_teacher_unchanged_threshold >= opt.gbcc_teacher_changed_threshold:
        raise ValueError("gbcc_teacher_unchanged_threshold must be smaller than gbcc_teacher_changed_threshold.")
    if opt.model_select_opt_weight < 0.0 or opt.model_select_sar_weight < 0.0:
        raise ValueError("model_select_opt_weight and model_select_sar_weight must be >= 0.")
    if (opt.model_select_opt_weight + opt.model_select_sar_weight) <= 0.0:
        raise ValueError("model_select_opt_weight + model_select_sar_weight must be > 0.")
    if opt.early_stop_patience < 1:
        raise ValueError("early_stop_patience must be >= 1.")
    if opt.num_workers < 0:
        raise ValueError("num_workers must be >= 0.")
    if opt.prefetch_factor < 1:
        raise ValueError("prefetch_factor must be >= 1.")
    if opt.profile_input_batch < 1:
        raise ValueError("profile_input_batch must be >= 1.")
    if opt.batch_size < 2:
        raise ValueError("batch_size must be >= 2 for current discriminator BatchNorm topology.")
    if opt.ucef_scale < 0.0:
        raise ValueError("ucef_scale must be >= 0.")
    if opt.racr_scale < 0.0:
        raise ValueError("racr_scale must be >= 0.")
    if opt.racr_base_suppress < 0.0:
        raise ValueError("racr_base_suppress must be >= 0.")
    if opt.lambda_racr_fp < 0.0:
        raise ValueError("lambda_racr_fp must be >= 0.")
    if opt.racr_start_epoch < 0:
        raise ValueError("racr_start_epoch must be >= 0.")
    if opt.racr_warmup_epochs < 1:
        raise ValueError("racr_warmup_epochs must be >= 1.")
    if opt.racr_fp_gamma <= 0.0:
        raise ValueError("racr_fp_gamma must be > 0.")
    if opt.lambda_barc_boundary < 0.0:
        raise ValueError("lambda_barc_boundary must be >= 0.")
    if opt.lambda_barc_region < 0.0:
        raise ValueError("lambda_barc_region must be >= 0.")
    if opt.barc_start_epoch < 0:
        raise ValueError("barc_start_epoch must be >= 0.")
    if opt.barc_warmup_epochs < 1:
        raise ValueError("barc_warmup_epochs must be >= 1.")
    if opt.barc_boundary_width < 1:
        raise ValueError("barc_boundary_width must be >= 1.")
    print(opt)
    if opt.use_gbcc:
        print(
            "[Legacy Notice] --use_gbcc is a compatibility flag; current training path uses SABR weighting and does not add GBCC loss."
        )
    if opt.use_sabr:
        print(
            f"[SABR] enabled start={opt.sabr_start_epoch} warmup={opt.sabr_warmup_epochs} "
            f"weights(ent/cd/reg)={opt.sabr_entropy_weight:.2f}/{opt.sabr_cd_weight:.2f}/{opt.sabr_reg_weight:.2f} "
            f"range=[{opt.sabr_min_weight:.2f}, {opt.sabr_max_weight:.2f}] temp={opt.sabr_temperature:.2f}"
        )
    else:
        print("[SABR] disabled, using static equal branch weights (0.5/0.5).")
    if opt.use_ucef:
        print(f"[UCEF] enabled in SNUNet_ECAM with residual scale={opt.ucef_scale:.3f}.")
    else:
        print("[UCEF] disabled; SNUNet_ECAM keeps the original CD head.")
    if opt.use_racr:
        print(
            f"[RACR] enabled scale={opt.racr_scale:.3f}, base_suppress={opt.racr_base_suppress:.3f}, "
            f"lambda_fp={opt.lambda_racr_fp:.4f}, start={opt.racr_start_epoch}, warmup={opt.racr_warmup_epochs}."
        )
    else:
        print("[RACR] disabled; no registration-aware CD refinement is applied.")
    if opt.use_barc:
        print(
            f"[BARC] enabled lambda_boundary={opt.lambda_barc_boundary:.4f}, "
            f"lambda_region={opt.lambda_barc_region:.4f}, start={opt.barc_start_epoch}, "
            f"warmup={opt.barc_warmup_epochs}, boundary_width={opt.barc_boundary_width}."
        )
    else:
        print("[BARC] disabled; CD loss keeps the original focal supervision.")

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    if opt.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested but not available.")
    if torch.cuda.is_available():
        if opt.gpu_id < 0 or opt.gpu_id >= torch.cuda.device_count():
            raise ValueError(f"Invalid gpu_id={opt.gpu_id}, available GPUs: 0..{torch.cuda.device_count() - 1}")
        device = torch.device(f"cuda:{opt.gpu_id}")
        cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    print(f"===> Using device: {device}")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if opt.run_name is not None and opt.run_name.strip():
        exp_name = f"{opt.dataset}_{opt.run_name.strip()}"
    else:
        exp_name = f"{opt.dataset}_HuiYanMTCDN_{timestamp}"
    result_dir = os.path.join("result", exp_name)
    image_dir = os.path.join(result_dir, "images")
    metrics_dir = os.path.join(result_dir, "metrics")
    checkpoint_dir = os.path.join("checkpoint", exp_name)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"===> Experiment: {exp_name}")

    print("===> Loading datasets")
    train_set = get_training_set(os.path.join("dataset", opt.dataset), opt.direction)
    test_set = get_test_set(os.path.join("dataset", opt.dataset), opt.direction)
    training_data_loader = build_loader(train_set, opt.batch_size, shuffle=True, drop_last=True, opt=opt)
    testing_data_loader = build_loader(test_set, opt.test_batch_size, shuffle=False, drop_last=True, opt=opt)

    print("===> Building models")
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
    D_opt.apply(weights_init_normal)
    D_sar.apply(weights_init_normal)

    pretrained_ok_opt2sar = False
    pretrained_ok_sar2opt = False
    if opt.use_pretrained:
        total_keys_opt2sar = len(G_opt2sar.state_dict())
        total_keys_sar2opt = len(G_sar2opt.state_dict())

        if os.path.exists(opt.pretrained_opt2sar_path):
            missing, unexpected = load_pretrained_generator(G_opt2sar, opt.pretrained_opt2sar_path, strict=False)
            loaded_ratio = 1.0 - (len(missing) / max(1, total_keys_opt2sar))
            pretrained_ok_opt2sar = loaded_ratio > 0.7
            print(
                f"Loaded G_opt2sar pretrained: missing={len(missing)}, unexpected={len(unexpected)}, "
                f"loaded_ratio={loaded_ratio:.3f}"
            )
        else:
            print(f"[WARN] pretrained not found, G_opt2sar random init: {opt.pretrained_opt2sar_path}")

        if os.path.exists(opt.pretrained_sar2opt_path):
            missing, unexpected = load_pretrained_generator(G_sar2opt, opt.pretrained_sar2opt_path, strict=False)
            loaded_ratio = 1.0 - (len(missing) / max(1, total_keys_sar2opt))
            pretrained_ok_sar2opt = loaded_ratio > 0.7
            print(
                f"Loaded G_sar2opt pretrained: missing={len(missing)}, unexpected={len(unexpected)}, "
                f"loaded_ratio={loaded_ratio:.3f}"
            )
        else:
            print(f"[WARN] pretrained not found, G_sar2opt random init: {opt.pretrained_sar2opt_path}")
    else:
        print("Skipping pretrained generator loading. Training from random initialization.")

    if not pretrained_ok_opt2sar:
        G_opt2sar.apply(weights_init_normal)
    if not pretrained_ok_sar2opt:
        G_sar2opt.apply(weights_init_normal)

    use_pretrained_schedule = pretrained_ok_opt2sar and pretrained_ok_sar2opt
    if use_pretrained_schedule:
        print("Using pretrained-aware freezing schedule.")
    else:
        print("Using no-pretrained adaptive schedule (no full freeze warmup).")

    G_opt2sar = G_opt2sar.to(device)
    G_sar2opt = G_sar2opt.to(device)
    D_opt = D_opt.to(device)
    D_sar = D_sar.to(device)

    if opt.profile_complexity or opt.profile_only:
        print("===> Profiling model complexity (Params / MACs / FLOPs)")
        profile_rows = []
        profile_bs = int(opt.profile_input_batch)
        dummy_a = torch.randn(profile_bs, opt.input_nc, opt.img_height, opt.img_width, device=device)
        dummy_b = torch.randn(profile_bs, opt.input_nc, opt.img_height, opt.img_width, device=device)
        profile_specs = [
            ("G_opt2sar", G_opt2sar, (dummy_a,)),
            ("G_sar2opt", G_sar2opt, (dummy_b,)),
            ("D_opt", D_opt, (dummy_a, dummy_b)),
            ("D_sar", D_sar, (dummy_b, dummy_a)),
        ]
        for model_name, model_obj, model_inputs in profile_specs:
            params_total, params_trainable = count_parameters(model_obj)
            try:
                macs = estimate_model_macs(model_obj, model_inputs)
            except Exception as exc:
                print(f"[WARN] MACs estimation failed for {model_name}: {exc}")
                macs = 0.0
            flops = macs * 2.0
            row = {
                "model": model_name,
                "params_total": int(params_total),
                "params_trainable": int(params_trainable),
                "macs": float(macs),
                "flops": float(flops),
                "params_total_fmt": format_count(params_total),
                "params_trainable_fmt": format_count(params_trainable),
                "macs_fmt": format_count(macs),
                "flops_fmt": format_count(flops),
            }
            profile_rows.append(row)
            print(
                f"[Profile] {model_name}: params={row['params_total_fmt']} "
                f"(trainable={row['params_trainable_fmt']}), MACs={row['macs_fmt']}, FLOPs={row['flops_fmt']}"
            )
        profile_txt_path = os.path.join(metrics_dir, "model_profile.txt")
        profile_csv_path = os.path.join(metrics_dir, "model_profile.csv")
        profile_note = (
            f"Input=(B={profile_bs}, C={opt.input_nc}, H={opt.img_height}, W={opt.img_width}); "
            "FLOPs are approximated as 2 x MACs."
        )
        save_model_profile(profile_rows, profile_txt_path, profile_csv_path, profile_note)
        print(f"Saved model profile: {profile_txt_path}")
        print(f"Saved model profile csv: {profile_csv_path}")
        if opt.profile_only:
            print("profile_only=True, exit after complexity profiling.")
            sys.exit(0)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        G_opt2sar = nn.DataParallel(G_opt2sar)
        G_sar2opt = nn.DataParallel(G_sar2opt)
        D_opt = nn.DataParallel(D_opt)
        D_sar = nn.DataParallel(D_sar)
        print(f"===> Multi-GPU enabled: {torch.cuda.device_count()} GPUs")

    criterion_gan = nn.MSELoss().to(device)
    criterion_cycle = nn.L1Loss().to(device)
    criterion_recon = nn.L1Loss().to(device)
    criterion_cd = FocalLoss(gamma=2, alpha=0.25).to(device)

    disc_out_shape = unwrap(D_opt).output_shape

    optimizer_G = torch.optim.Adam(
        itertools.chain(G_opt2sar.parameters(), G_sar2opt.parameters()),
        lr=opt.G_lr,
        betas=(opt.b1, opt.b2),
    )
    optimizer_D_opt = torch.optim.Adam(D_opt.parameters(), lr=opt.D_lr, betas=(opt.b1, opt.b2))
    optimizer_D_sar = torch.optim.Adam(D_sar.parameters(), lr=opt.D_lr, betas=(opt.b1, opt.b2))

    decay_start = min(opt.niter_decay, max(1, opt.niter - 1))
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.niter, opt.epoch, decay_start).step
    )
    lr_scheduler_D_opt = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_opt, lr_lambda=LambdaLR(opt.niter, opt.epoch, decay_start).step
    )
    lr_scheduler_D_sar = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_sar, lr_lambda=LambdaLR(opt.niter, opt.epoch, decay_start).step
    )

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()
    real_A_buffer = ReplayBuffer()
    real_B_buffer = ReplayBuffer()
    lbl_buffer = ReplayBuffer()

    def sample_images(real_A, real_B, epoch_id):
        if not opt.save_generated_images:
            return
        G_opt2sar.eval()
        G_sar2opt.eval()
        with torch.no_grad():
            fake_B = G_opt2sar(real_A)
            fake_A = G_sar2opt(real_B)
            grid_real_A = make_grid(to_zero_one(real_A), nrow=5, normalize=False)
            grid_fake_B = make_grid(to_zero_one(fake_B), nrow=5, normalize=False)
            grid_real_B = make_grid(to_zero_one(real_B), nrow=5, normalize=False)
            grid_fake_A = make_grid(to_zero_one(fake_A), nrow=5, normalize=False)
            image_grid = torch.cat((grid_real_A, grid_fake_B, grid_real_B, grid_fake_A), 1)
            save_image(image_grid, os.path.join(image_dir, f"{epoch_id}.png"), normalize=False)

    best_score = -1.0
    best_epoch = -1
    best_f1_opt = 0.0
    best_f1_sar = 0.0
    best_iou_opt = 0.0
    best_iou_sar = 0.0
    no_optim = 0
    prev_time = time.time()
    epoch_rows = []
    epoch_metrics_path = os.path.join(metrics_dir, "epoch_metrics.csv")
    epoch_curve_path = os.path.join(metrics_dir, "training_curves.png")
    epoch_fieldnames = [
        "epoch",
        "stage",
        "sar_region_mode",
        "train_time_s",
        "eval_time_s",
        "epoch_time_s",
        "peak_mem_mb",
        "samples_per_sec",
        "batches_per_sec",
        "train_g_loss",
        "train_d_loss",
        "train_adv_loss",
        "train_recon_loss",
        "train_cycle_loss",
        "train_cd_opt_loss",
        "train_cd_sar_loss",
        "train_reg_opt_loss",
        "train_reg_sar_loss",
        "train_racr_fp_opt_loss",
        "train_racr_fp_sar_loss",
        "train_barc_boundary_opt_loss",
        "train_barc_boundary_sar_loss",
        "train_barc_region_opt_loss",
        "train_barc_region_sar_loss",
        "lambda_cgdr_opt_eff",
        "lambda_cgdr_sar_eff",
        "lambda_racr_fp_eff",
        "lambda_barc_boundary_eff",
        "lambda_barc_region_eff",
        "sabr_scale",
        "sabr_w_opt",
        "sabr_w_sar",
        "sabr_r_opt",
        "sabr_r_sar",
        "beta_cd",
        "opt_oa",
        "opt_precision",
        "opt_recall",
        "opt_f1",
        "opt_iou",
        "sar_oa",
        "sar_precision",
        "sar_recall",
        "sar_f1",
        "sar_iou",
        "selection_score",
        "best_score",
        "best_epoch",
        "no_optim",
    ]
    if opt.export_epoch_metrics:
        with open(epoch_metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=epoch_fieldnames)
            writer.writeheader()
        print(f"Epoch metrics will be saved to: {epoch_metrics_path}")

    for epoch in range(opt.epoch_count, opt.niter + 1):
        unwrap(D_opt).set_cgdr_region_mode(opt.opt_cgdr_region_mode)
        if opt.sar_scatter_start_epoch >= 0:
            sar_region_mode = "sar_scatter" if epoch >= opt.sar_scatter_start_epoch else "correlation"
        else:
            sar_region_mode = opt.sar_cgdr_region_mode
        unwrap(D_sar).set_cgdr_region_mode(sar_region_mode)

        if use_pretrained_schedule:
            if epoch < opt.freeze_stage0_epochs:
                stage = "all_frozen"
            elif epoch < opt.partial_finetune_until:
                stage = "last5"
            else:
                stage = "all_trainable"
        else:
            if epoch < max(2, opt.freeze_stage0_epochs):
                stage = "last5"
            else:
                stage = "all_trainable"
        g1_trainable, g1_total = set_generator_train_stage(
            G_opt2sar, stage=stage, freeze_first_n=opt.freeze_generator_layers
        )
        g2_trainable, g2_total = set_generator_train_stage(
            G_sar2opt, stage=stage, freeze_first_n=opt.freeze_generator_layers
        )
        print(
            f"\n[Epoch {epoch}] generator_stage={stage} G_opt2sar={g1_trainable}/{g1_total} "
            f"G_sar2opt={g2_trainable}/{g2_total} sar_region_mode={sar_region_mode}"
        )

        epoch_start_t = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        run_g_loss = 0.0
        run_d_loss = 0.0
        run_adv_loss = 0.0
        run_recon_loss = 0.0
        run_cycle_loss = 0.0
        run_cd_opt_loss = 0.0
        run_cd_sar_loss = 0.0
        run_reg_opt_loss = 0.0
        run_reg_sar_loss = 0.0
        run_racr_fp_opt_loss = 0.0
        run_racr_fp_sar_loss = 0.0
        run_barc_boundary_opt_loss = 0.0
        run_barc_boundary_sar_loss = 0.0
        run_barc_region_opt_loss = 0.0
        run_barc_region_sar_loss = 0.0
        run_sabr_w_opt = 0.0
        run_sabr_w_sar = 0.0
        last_lambda_cgdr_opt_eff = 0.0
        last_lambda_cgdr_sar_eff = 0.0
        last_lambda_racr_fp_eff = 0.0
        last_lambda_barc_boundary_eff = 0.0
        last_lambda_barc_region_eff = 0.0
        last_sabr_scale = 0.0
        last_sabr_aux = {
            "r_opt": 0.5,
            "r_sar": 0.5,
            "w_opt": 0.5,
            "w_sar": 0.5,
        }
        last_beta_cd = 0.0

        A_TP = A_TN = A_FP = A_FN = 0.0
        B_TP = B_TN = B_FP = B_FN = 0.0

        for iteration, batch in enumerate(training_data_loader, 1):
            real_A = to_neg_one_pos_one(batch[0].to(device, dtype=torch.float))
            real_B = to_neg_one_pos_one(batch[1].to(device, dtype=torch.float))
            lbl = batch[2].to(device, dtype=torch.long)

            G_opt2sar.train()
            G_sar2opt.train()
            D_opt.train()
            D_sar.train()

            if stage == "all_frozen":
                with torch.no_grad():
                    fake_B = G_opt2sar(real_A)
                    fake_A = G_sar2opt(real_B)
            else:
                fake_B = G_opt2sar(real_A)
                fake_A = G_sar2opt(real_B)
            valid = Variable(torch.ones((real_A.shape[0], *disc_out_shape), device=device), requires_grad=False)
            fake = Variable(torch.zeros((real_A.shape[0], *disc_out_shape), device=device), requires_grad=False)

            loss_recon_A = criterion_recon(G_sar2opt(real_A), real_A)
            loss_recon_B = criterion_recon(G_opt2sar(real_B), real_B)
            loss_recon = (loss_recon_A + loss_recon_B) / 2.0

            loss_gan_opt2sar = criterion_gan(D_sar(real_B, fake_B)[1], valid)
            loss_gan_sar2opt = criterion_gan(D_opt(real_A, fake_A)[1], valid)
            loss_adv = (loss_gan_opt2sar + loss_gan_sar2opt) / 2.0

            recov_A = G_sar2opt(fake_B)
            recov_B = G_opt2sar(fake_A)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2.0

            loss_G = loss_adv + opt.lambda_id * loss_recon + opt.lambda_cyc * loss_cycle

            if stage != "all_frozen":
                optimizer_G.zero_grad()
                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(G_opt2sar.parameters(), G_sar2opt.parameters()),
                    max_norm=5.0,
                )
                optimizer_G.step()

            optimizer_D_opt.zero_grad()
            optimizer_D_sar.zero_grad()

            fake_A_ = fake_A_buffer.push_and_pop(fake_A.detach())
            fake_B_ = fake_B_buffer.push_and_pop(fake_B.detach())
            real_A_ = real_A_buffer.push_and_pop(real_A.detach())
            real_B_ = real_B_buffer.push_and_pop(real_B.detach())
            lbl_ = lbl_buffer.push_and_pop(lbl.detach())
            unchanged_mask = (lbl_ == 0).float().unsqueeze(1)
            opt_mask = unchanged_mask if opt.opt_reg_use_unchanged_mask else None
            sar_mask = unchanged_mask if opt.sar_reg_use_unchanged_mask else None

            if opt.use_racr:
                real_A_logit, fake_A_logit, output_A, loss_reg_opt, racr_aux_opt = D_opt(
                    real_A_, fake_A_, return_registration=True, valid_mask=opt_mask, return_cd_aux=True
                )
                real_B_logit, fake_B_logit, output_B, loss_reg_sar, racr_aux_sar = D_sar(
                    real_B_, fake_B_, return_registration=True, valid_mask=sar_mask, return_cd_aux=True
                )
            else:
                real_A_logit, fake_A_logit, output_A, loss_reg_opt = D_opt(
                    real_A_, fake_A_, return_registration=True, valid_mask=opt_mask
                )
                real_B_logit, fake_B_logit, output_B, loss_reg_sar = D_sar(
                    real_B_, fake_B_, return_registration=True, valid_mask=sar_mask
                )
                racr_aux_opt = None
                racr_aux_sar = None

            loss_D_opt_adv = criterion_gan(real_A_logit, valid) + criterion_gan(fake_A_logit, fake)
            loss_D_sar_adv = criterion_gan(real_B_logit, valid) + criterion_gan(fake_B_logit, fake)
            loss_CD_opt = criterion_cd(output_A, lbl_)
            loss_CD_sar = criterion_cd(output_B, lbl_)
            loss_racr_fp_opt = compute_racr_fp_loss(output_A, lbl_, racr_aux_opt, gamma=opt.racr_fp_gamma)
            loss_racr_fp_sar = compute_racr_fp_loss(output_B, lbl_, racr_aux_sar, gamma=opt.racr_fp_gamma)
            if opt.use_barc:
                loss_barc_boundary_opt, loss_barc_region_opt = compute_barc_losses(
                    output_A, lbl_, boundary_width=opt.barc_boundary_width
                )
                loss_barc_boundary_sar, loss_barc_region_sar = compute_barc_losses(
                    output_B, lbl_, boundary_width=opt.barc_boundary_width
                )
            else:
                loss_barc_boundary_opt = output_A.new_zeros(())
                loss_barc_region_opt = output_A.new_zeros(())
                loss_barc_boundary_sar = output_B.new_zeros(())
                loss_barc_region_sar = output_B.new_zeros(())
            sabr_mask = unchanged_mask if opt.sabr_use_unchanged_mask else None
            if opt.use_sabr and epoch >= opt.sabr_start_epoch:
                w_opt_sabr, w_sar_sabr, sabr_aux = compute_sabr_weights(
                    output_A,
                    output_B,
                    loss_CD_opt,
                    loss_CD_sar,
                    loss_reg_opt=loss_reg_opt,
                    loss_reg_sar=loss_reg_sar,
                    entropy_weight=opt.sabr_entropy_weight,
                    cd_weight=opt.sabr_cd_weight,
                    reg_weight=opt.sabr_reg_weight,
                    temperature=opt.sabr_temperature,
                    min_weight=opt.sabr_min_weight,
                    max_weight=opt.sabr_max_weight,
                    valid_mask=sabr_mask,
                )
                sabr_scale = min(
                    1.0,
                    float(epoch - opt.sabr_start_epoch + 1) / max(1.0, float(opt.sabr_warmup_epochs)),
                )
            else:
                w_opt_sabr = output_A.new_tensor(0.5)
                w_sar_sabr = output_A.new_tensor(0.5)
                sabr_aux = {"r_opt": 0.5, "r_sar": 0.5, "w_opt": 0.5, "w_sar": 0.5}
                sabr_scale = 0.0

            if use_pretrained_schedule:
                beta_cd = 0.0 if epoch < opt.freeze_stage0_epochs else (0.5 if epoch < opt.partial_finetune_until else 2.0)
            else:
                beta_cd = 0.2 if epoch < max(2, opt.freeze_stage0_epochs) else 1.0
            cgdr_lambda_scale_opt = min(1.0, float(epoch) / max(1.0, float(opt.cgdr_warmup_epochs_opt)))
            cgdr_lambda_scale_sar = min(1.0, float(epoch) / max(1.0, float(opt.cgdr_warmup_epochs_sar)))
            lambda_cgdr_opt_eff = opt.lambda_cgdr_opt * cgdr_lambda_scale_opt
            lambda_cgdr_sar_eff = opt.lambda_cgdr_sar * cgdr_lambda_scale_sar
            if opt.use_racr and epoch >= opt.racr_start_epoch:
                racr_lambda_scale = min(
                    1.0,
                    float(epoch - opt.racr_start_epoch + 1) / max(1.0, float(opt.racr_warmup_epochs)),
                )
            else:
                racr_lambda_scale = 0.0
            lambda_racr_fp_eff = opt.lambda_racr_fp * racr_lambda_scale
            if opt.use_barc and epoch >= opt.barc_start_epoch:
                barc_lambda_scale = min(
                    1.0,
                    float(epoch - opt.barc_start_epoch + 1) / max(1.0, float(opt.barc_warmup_epochs)),
                )
            else:
                barc_lambda_scale = 0.0
            lambda_barc_boundary_eff = opt.lambda_barc_boundary * barc_lambda_scale
            lambda_barc_region_eff = opt.lambda_barc_region * barc_lambda_scale
            w_opt_eff = (1.0 - sabr_scale) * 0.5 + sabr_scale * float(w_opt_sabr.detach().cpu())
            w_sar_eff = (1.0 - sabr_scale) * 0.5 + sabr_scale * float(w_sar_sabr.detach().cpu())
            w_norm = max(w_opt_eff + w_sar_eff, 1e-8)
            w_opt_eff /= w_norm
            w_sar_eff /= w_norm
            loss_D_opt = (
                loss_D_opt_adv
                + beta_cd * loss_CD_opt
                + lambda_cgdr_opt_eff * loss_reg_opt
                + lambda_racr_fp_eff * loss_racr_fp_opt
                + lambda_barc_boundary_eff * loss_barc_boundary_opt
                + lambda_barc_region_eff * loss_barc_region_opt
            )
            loss_D_sar = (
                loss_D_sar_adv
                + beta_cd * loss_CD_sar
                + lambda_cgdr_sar_eff * loss_reg_sar
                + lambda_racr_fp_eff * loss_racr_fp_sar
                + lambda_barc_boundary_eff * loss_barc_boundary_sar
                + lambda_barc_region_eff * loss_barc_region_sar
            )
            loss_D = w_opt_eff * loss_D_opt + w_sar_eff * loss_D_sar
            loss_D_total = loss_D
            loss_reg = (loss_reg_opt + loss_reg_sar) / 2.0

            run_g_loss += loss_G.item()
            run_d_loss += loss_D.item()
            run_adv_loss += loss_adv.item()
            run_recon_loss += loss_recon.item()
            run_cycle_loss += loss_cycle.item()
            run_cd_opt_loss += loss_CD_opt.item()
            run_cd_sar_loss += loss_CD_sar.item()
            run_reg_opt_loss += loss_reg_opt.item()
            run_reg_sar_loss += loss_reg_sar.item()
            run_racr_fp_opt_loss += loss_racr_fp_opt.item()
            run_racr_fp_sar_loss += loss_racr_fp_sar.item()
            run_barc_boundary_opt_loss += loss_barc_boundary_opt.item()
            run_barc_boundary_sar_loss += loss_barc_boundary_sar.item()
            run_barc_region_opt_loss += loss_barc_region_opt.item()
            run_barc_region_sar_loss += loss_barc_region_sar.item()
            run_sabr_w_opt += w_opt_eff
            run_sabr_w_sar += w_sar_eff
            last_lambda_cgdr_opt_eff = float(lambda_cgdr_opt_eff)
            last_lambda_cgdr_sar_eff = float(lambda_cgdr_sar_eff)
            last_lambda_racr_fp_eff = float(lambda_racr_fp_eff)
            last_lambda_barc_boundary_eff = float(lambda_barc_boundary_eff)
            last_lambda_barc_region_eff = float(lambda_barc_region_eff)
            last_sabr_scale = float(sabr_scale)
            last_sabr_aux = sabr_aux
            last_beta_cd = float(beta_cd)

            loss_D_total.backward()
            optimizer_D_opt.step()
            optimizer_D_sar.step()

            batches_done = epoch
            batches_left = opt.niter * len(training_data_loader) - batches_done
            time_left = dt.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            if batches_done % opt.sample_interval == 0:
                sample_images(real_A, real_B, batches_done)

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %.6f, cgdr(o/s): %.6f/%.6f, lambda(o/s): %.5f/%.5f, racr_fp(o/s): %.6f/%.6f, lambda: %.5f, barc(b/r): %.6f/%.6f, lambda(b/r): %.5f/%.5f, sabr(w o/s): %.4f/%.4f, scale: %.4f] [G loss: %.6f, adv: %.6f, recon: %.6f, cycle: %.6f] ETA: %s"
                % (
                    epoch,
                    opt.niter,
                    iteration,
                    len(training_data_loader),
                    loss_D.item(),
                    loss_reg_opt.item(),
                    loss_reg_sar.item(),
                    lambda_cgdr_opt_eff,
                    lambda_cgdr_sar_eff,
                    loss_racr_fp_opt.item(),
                    loss_racr_fp_sar.item(),
                    lambda_racr_fp_eff,
                    0.5 * (loss_barc_boundary_opt.item() + loss_barc_boundary_sar.item()),
                    0.5 * (loss_barc_region_opt.item() + loss_barc_region_sar.item()),
                    lambda_barc_boundary_eff,
                    lambda_barc_region_eff,
                    w_opt_eff,
                    w_sar_eff,
                    sabr_scale,
                    loss_G.item(),
                    loss_adv.item(),
                    loss_recon.item(),
                    loss_cycle.item(),
                    time_left,
                )
            )

        train_time_s = time.time() - epoch_start_t
        eval_start_t = time.time()
        G_opt2sar.eval()
        G_sar2opt.eval()
        D_opt.eval()
        D_sar.eval()
        with torch.no_grad():
            for _, batch in enumerate(testing_data_loader, 1):
                real_A = to_neg_one_pos_one(batch[0].to(device, dtype=torch.float))
                real_B = to_neg_one_pos_one(batch[1].to(device, dtype=torch.float))
                lbl = batch[2].to(device, dtype=torch.long)

                fake_B = G_opt2sar(real_A)
                fake_A = G_sar2opt(real_B)
                _, _, output_A = D_opt(real_A, fake_A)
                _, _, output_B = D_sar(real_B, fake_B)

                pred_A = torch.argmax(output_A, 1).squeeze()
                pred_B = torch.argmax(output_B, 1).squeeze()

                A_TP += ((pred_A == 1).long() & (lbl == 1).long()).float().sum().item()
                A_TN += ((pred_A == 0).long() & (lbl == 0).long()).float().sum().item()
                A_FP += ((pred_A == 1).long() & (lbl == 0).long()).float().sum().item()
                A_FN += ((pred_A == 0).long() & (lbl == 1).long()).float().sum().item()

                B_TP += ((pred_B == 1).long() & (lbl == 1).long()).float().sum().item()
                B_TN += ((pred_B == 0).long() & (lbl == 0).long()).float().sum().item()
                B_FP += ((pred_B == 1).long() & (lbl == 0).long()).float().sum().item()
                B_FN += ((pred_B == 0).long() & (lbl == 1).long()).float().sum().item()

        oa_A, p_A, r_A, f1_A, iou_A = compute_metrics(A_TP, A_TN, A_FP, A_FN)
        oa_B, p_B, r_B, f1_B, iou_B = compute_metrics(B_TP, B_TN, B_FP, B_FN)
        eval_time_s = time.time() - eval_start_t
        epoch_time_s = train_time_s + eval_time_s
        num_batches = max(1, len(training_data_loader))
        avg_g_loss = run_g_loss / num_batches
        avg_d_loss = run_d_loss / num_batches
        avg_adv_loss = run_adv_loss / num_batches
        avg_recon_loss = run_recon_loss / num_batches
        avg_cycle_loss = run_cycle_loss / num_batches
        avg_cd_opt_loss = run_cd_opt_loss / num_batches
        avg_cd_sar_loss = run_cd_sar_loss / num_batches
        avg_reg_opt_loss = run_reg_opt_loss / num_batches
        avg_reg_sar_loss = run_reg_sar_loss / num_batches
        avg_racr_fp_opt_loss = run_racr_fp_opt_loss / num_batches
        avg_racr_fp_sar_loss = run_racr_fp_sar_loss / num_batches
        avg_barc_boundary_opt_loss = run_barc_boundary_opt_loss / num_batches
        avg_barc_boundary_sar_loss = run_barc_boundary_sar_loss / num_batches
        avg_barc_region_opt_loss = run_barc_region_opt_loss / num_batches
        avg_barc_region_sar_loss = run_barc_region_sar_loss / num_batches
        avg_sabr_w_opt = run_sabr_w_opt / num_batches
        avg_sabr_w_sar = run_sabr_w_sar / num_batches
        samples_per_sec = (num_batches * opt.batch_size) / max(train_time_s, 1e-8)
        batches_per_sec = num_batches / max(train_time_s, 1e-8)
        peak_mem_mb = (
            float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else 0.0
        )
        print()
        print(f"Optical -> OA: {oa_A:.6f} Precision: {p_A:.6f} Recall: {r_A:.6f} F1: {f1_A:.6f} IoU: {iou_A:.6f}")
        print(f"SAR     -> OA: {oa_B:.6f} Precision: {p_B:.6f} Recall: {r_B:.6f} F1: {f1_B:.6f} IoU: {iou_B:.6f}")
        print(
            f"Epoch time: total={epoch_time_s:.2f}s (train={train_time_s:.2f}s, eval={eval_time_s:.2f}s), "
            f"speed={samples_per_sec:.2f} samples/s, peak_mem={peak_mem_mb:.1f} MB"
        )
        print(
            f"SABR -> scale: {last_sabr_scale:.4f}, r(opt/sar): {last_sabr_aux['r_opt']:.4f}/{last_sabr_aux['r_sar']:.4f}, "
            f"w(opt/sar): {avg_sabr_w_opt:.4f}/{avg_sabr_w_sar:.4f}"
        )
        if opt.use_racr:
            print(
                f"RACR -> lambda_fp: {last_lambda_racr_fp_eff:.5f}, "
                f"fp_loss(opt/sar): {avg_racr_fp_opt_loss:.6f}/{avg_racr_fp_sar_loss:.6f}"
            )
        if opt.use_barc:
            print(
                f"BARC -> lambda_boundary/region: {last_lambda_barc_boundary_eff:.5f}/{last_lambda_barc_region_eff:.5f}, "
                f"boundary(opt/sar): {avg_barc_boundary_opt_loss:.6f}/{avg_barc_boundary_sar_loss:.6f}, "
                f"region(opt/sar): {avg_barc_region_opt_loss:.6f}/{avg_barc_region_sar_loss:.6f}"
            )
        current_score = compute_selection_score(opt, f1_A, f1_B, iou_A, iou_B)
        print(
            f"Selection metric ({opt.model_select_metric}): {current_score:.6f} "
            f"(w_opt={opt.model_select_opt_weight:.2f}, w_sar={opt.model_select_sar_weight:.2f})"
        )

        if current_score <= best_score:
            no_optim += 1
        else:
            no_optim = 0
            best_score = current_score
            best_epoch = epoch
            best_f1_opt = f1_A
            best_f1_sar = f1_B
            best_iou_opt = iou_A
            best_iou_sar = iou_B
            if epoch != 0:
                print("Saving best model by selection metric.")
                torch.save(unwrap(G_opt2sar).state_dict(), os.path.join(checkpoint_dir, "G_opt2sar_best.pth"))
                torch.save(unwrap(G_sar2opt).state_dict(), os.path.join(checkpoint_dir, "G_sar2opt_best.pth"))
                torch.save(unwrap(D_opt).state_dict(), os.path.join(checkpoint_dir, "D_opt_best.pth"))
                torch.save(unwrap(D_sar).state_dict(), os.path.join(checkpoint_dir, "D_sar_best.pth"))

        epoch_row = {
            "epoch": int(epoch),
            "stage": stage,
            "sar_region_mode": sar_region_mode,
            "train_time_s": float(train_time_s),
            "eval_time_s": float(eval_time_s),
            "epoch_time_s": float(epoch_time_s),
            "peak_mem_mb": float(peak_mem_mb),
            "samples_per_sec": float(samples_per_sec),
            "batches_per_sec": float(batches_per_sec),
            "train_g_loss": float(avg_g_loss),
            "train_d_loss": float(avg_d_loss),
            "train_adv_loss": float(avg_adv_loss),
            "train_recon_loss": float(avg_recon_loss),
            "train_cycle_loss": float(avg_cycle_loss),
            "train_cd_opt_loss": float(avg_cd_opt_loss),
            "train_cd_sar_loss": float(avg_cd_sar_loss),
            "train_reg_opt_loss": float(avg_reg_opt_loss),
            "train_reg_sar_loss": float(avg_reg_sar_loss),
            "train_racr_fp_opt_loss": float(avg_racr_fp_opt_loss),
            "train_racr_fp_sar_loss": float(avg_racr_fp_sar_loss),
            "train_barc_boundary_opt_loss": float(avg_barc_boundary_opt_loss),
            "train_barc_boundary_sar_loss": float(avg_barc_boundary_sar_loss),
            "train_barc_region_opt_loss": float(avg_barc_region_opt_loss),
            "train_barc_region_sar_loss": float(avg_barc_region_sar_loss),
            "lambda_cgdr_opt_eff": float(last_lambda_cgdr_opt_eff),
            "lambda_cgdr_sar_eff": float(last_lambda_cgdr_sar_eff),
            "lambda_racr_fp_eff": float(last_lambda_racr_fp_eff),
            "lambda_barc_boundary_eff": float(last_lambda_barc_boundary_eff),
            "lambda_barc_region_eff": float(last_lambda_barc_region_eff),
            "sabr_scale": float(last_sabr_scale),
            "sabr_w_opt": float(avg_sabr_w_opt),
            "sabr_w_sar": float(avg_sabr_w_sar),
            "sabr_r_opt": float(last_sabr_aux["r_opt"]),
            "sabr_r_sar": float(last_sabr_aux["r_sar"]),
            "beta_cd": float(last_beta_cd),
            "opt_oa": float(oa_A),
            "opt_precision": float(p_A),
            "opt_recall": float(r_A),
            "opt_f1": float(f1_A),
            "opt_iou": float(iou_A),
            "sar_oa": float(oa_B),
            "sar_precision": float(p_B),
            "sar_recall": float(r_B),
            "sar_f1": float(f1_B),
            "sar_iou": float(iou_B),
            "selection_score": float(current_score),
            "best_score": float(best_score),
            "best_epoch": int(best_epoch),
            "no_optim": int(no_optim),
        }
        epoch_rows.append(epoch_row)
        if opt.export_epoch_metrics:
            with open(epoch_metrics_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=epoch_fieldnames)
                writer.writerow(epoch_row)

        if no_optim > opt.early_stop_patience and epoch >= opt.early_stop_min_epoch:
            print(f"Early stop at epoch {epoch}")
            break

        if epoch != 0 and opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            torch.save(unwrap(G_opt2sar).state_dict(), os.path.join(checkpoint_dir, f"G_opt2sar_{epoch}.pth"))
            torch.save(unwrap(G_sar2opt).state_dict(), os.path.join(checkpoint_dir, f"G_sar2opt_{epoch}.pth"))
            torch.save(unwrap(D_opt).state_dict(), os.path.join(checkpoint_dir, f"D_opt_{epoch}.pth"))
            torch.save(unwrap(D_sar).state_dict(), os.path.join(checkpoint_dir, f"D_sar_{epoch}.pth"))

        lr_scheduler_G.step()
        lr_scheduler_D_opt.step()
        lr_scheduler_D_sar.step()
        print(
            f"Best({opt.model_select_metric})={best_score:.6f} at epoch {best_epoch}; "
            f"F1(opt/sar)={best_f1_opt:.6f}/{best_f1_sar:.6f}, "
            f"IoU(opt/sar)={best_iou_opt:.6f}/{best_iou_sar:.6f}"
        )

    if opt.plot_epoch_curves:
        maybe_plot_epoch_curves(epoch_rows, epoch_curve_path)
        print(f"Saved training curves: {epoch_curve_path}")
