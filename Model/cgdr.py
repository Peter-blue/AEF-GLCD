import torch
import torch.nn as nn
import torch.nn.functional as F


class LeeSpeckleSuppressor(nn.Module):
    """Lee-style local statistics filter used before correlation estimation."""

    def __init__(self, window_size=5, eps=1e-6):
        super(LeeSpeckleSuppressor, self).__init__()
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        self.window_size = window_size
        self.eps = eps

    def forward(self, x):
        padding = self.window_size // 2
        local_mean = F.avg_pool2d(x, self.window_size, stride=1, padding=padding)
        local_sq_mean = F.avg_pool2d(x * x, self.window_size, stride=1, padding=padding)
        local_var = (local_sq_mean - local_mean * local_mean).clamp_min(0.0)
        noise_var = local_var.flatten(2).mean(dim=-1).view(x.size(0), x.size(1), 1, 1)
        lee_weight = local_var / (local_var + noise_var + self.eps)
        return local_mean + lee_weight * (x - local_mean)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvNormAct, self).__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CorrelationGuidedDeformableRegistration(nn.Module):
    """
    CGDR aligns the second image to the first image before change detection.

    High-correlation regions use a hard-gated fine residual flow. Low-correlation
    regions keep a coarse, low-resolution flow estimated from speckle-suppressed
    features, which makes SAR speckle less likely to dominate matching.
    """

    def __init__(
        self,
        in_channels=3,
        feature_channels=32,
        corr_window=5,
        high_corr_threshold=0.35,
        max_coarse_flow=8.0,
        max_fine_flow=2.0,
        gate_temperature=12.0,
        coarse_scale=4,
        blend_floor=0.15,
        blend_corr_start=0.0,
        region_mode="correlation",
        scatter_threshold=0.55,
        adaptive_gate_alpha=0.25,
        target_high_ratio=0.30,
        min_high_ratio=0.10,
        max_high_ratio=0.60,
        residual_suppress=0.35,
        low_conf_flow_scale=0.55,
        residual_conf_temperature=6.0,
        mask_alignment_with_valid=True,
        change_preserve_strength=0.0,
        change_preserve_threshold=0.35,
        change_preserve_temperature=10.0,
        use_speckle_filter=True,
        use_coarse_fine_split=True,
        eps=1e-6,
    ):
        super(CorrelationGuidedDeformableRegistration, self).__init__()
        self.high_corr_threshold = high_corr_threshold
        self.max_coarse_flow = max_coarse_flow
        self.max_fine_flow = max_fine_flow
        self.gate_temperature = gate_temperature
        self.coarse_scale = coarse_scale
        self.corr_window = corr_window
        self.blend_floor = blend_floor
        self.blend_corr_start = blend_corr_start
        self.region_mode = region_mode
        self.scatter_threshold = scatter_threshold
        self.adaptive_gate_alpha = adaptive_gate_alpha
        self.target_high_ratio = target_high_ratio
        self.min_high_ratio = min_high_ratio
        self.max_high_ratio = max_high_ratio
        self.residual_suppress = residual_suppress
        self.low_conf_flow_scale = low_conf_flow_scale
        self.residual_conf_temperature = residual_conf_temperature
        self.mask_alignment_with_valid = mask_alignment_with_valid
        self.change_preserve_strength = change_preserve_strength
        self.change_preserve_threshold = change_preserve_threshold
        self.change_preserve_temperature = change_preserve_temperature
        self.use_speckle_filter = use_speckle_filter
        self.use_coarse_fine_split = use_coarse_fine_split
        self.eps = eps

        self.speckle_filter = LeeSpeckleSuppressor(window_size=5, eps=eps)
        self.feature = nn.Sequential(
            ConvNormAct(in_channels, feature_channels),
            ConvNormAct(feature_channels, feature_channels),
        )
        self.semantic = nn.Sequential(
            ConvNormAct(in_channels, feature_channels),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
        )

        coarse_in = feature_channels * 3 + 1
        fine_in = feature_channels * 3 + 2
        self.coarse_head = nn.Sequential(
            ConvNormAct(coarse_in, feature_channels),
            ConvNormAct(feature_channels, feature_channels),
            nn.Conv2d(feature_channels, 2, kernel_size=3, padding=1),
        )
        self.fine_head = nn.Sequential(
            ConvNormAct(fine_in, feature_channels),
            ConvNormAct(feature_channels, feature_channels),
            nn.Conv2d(feature_channels, 2, kernel_size=3, padding=1),
        )
        self.reset_flow_heads()

    def reset_flow_heads(self):
        nn.init.zeros_(self.coarse_head[-1].weight)
        nn.init.zeros_(self.coarse_head[-1].bias)
        nn.init.zeros_(self.fine_head[-1].weight)
        nn.init.zeros_(self.fine_head[-1].bias)

    def forward(self, fixed, moving, return_aux=False, compute_loss=True, valid_mask=None):
        flow, aligned, aux = self._estimate_flow(fixed, moving)
        valid_mask = self._prepare_mask(valid_mask, aligned)
        if valid_mask is not None and self.mask_alignment_with_valid:
            aligned = valid_mask * aligned + (1.0 - valid_mask) * moving
            flow = valid_mask * flow

        if not compute_loss:
            reg_loss = fixed.new_zeros(())
            if return_aux:
                aux = dict(aux)
                aux.update({"aligned": aligned, "flow": flow})
                return aligned, reg_loss, aux
            return aligned, reg_loss

        backward_flow, _, _ = self._estimate_flow(moving, fixed)
        warped_backward = self._warp(backward_flow, flow)
        reg_mask = valid_mask
        inverse_loss = self._masked_l1(flow + warped_backward, torch.zeros_like(flow), reg_mask)
        smooth_loss = self._edge_aware_smoothness(flow, fixed, reg_mask)

        fixed_sem = self.semantic(self._filter_speckle(fixed))
        aligned_sem = self.semantic(self._filter_speckle(aligned))
        high_gate = aux["high_gate"].detach()
        high_mask = high_gate if reg_mask is None else high_gate * reg_mask
        low_mask = aux["low_gate"].detach() if reg_mask is None else aux["low_gate"].detach() * reg_mask

        semantic_loss = self._masked_l1(fixed_sem, aligned_sem, high_mask)

        aligned_feat = self.feature(self._filter_speckle(aligned))
        corr_after = self._local_correlation(aux["fixed_feat"], aligned_feat)
        correlation_loss = self._masked_l1(1.0 - corr_after, torch.zeros_like(corr_after), high_mask)
        # Keep low-correlation regions stable in a low-frequency sense.
        fixed_lp = F.avg_pool2d(self._filter_speckle(fixed), kernel_size=7, stride=1, padding=3)
        aligned_lp = F.avg_pool2d(self._filter_speckle(aligned), kernel_size=7, stride=1, padding=3)
        coarse_consistency_loss = self._masked_l1(fixed_lp, aligned_lp, low_mask)

        reg_loss = (
            0.32 * inverse_loss
            + 0.23 * smooth_loss
            + 0.20 * semantic_loss
            + 0.15 * correlation_loss
            + 0.10 * coarse_consistency_loss
        )

        if return_aux:
            aux = dict(aux)
            aux.update(
                {
                    "aligned": aligned,
                    "flow": flow,
                    "backward_flow": backward_flow,
                    "corr_after": corr_after,
                    "inverse_loss": inverse_loss.detach(),
                    "smooth_loss": smooth_loss.detach(),
                    "semantic_loss": semantic_loss.detach(),
                    "correlation_loss": correlation_loss.detach(),
                    "coarse_consistency_loss": coarse_consistency_loss.detach(),
                }
            )
            return aligned, reg_loss, aux
        return aligned, reg_loss

    def _estimate_flow(self, fixed, moving):
        fixed_smooth = self._filter_speckle(fixed)
        moving_smooth = self._filter_speckle(moving)
        fixed_feat = self.feature(fixed_smooth)
        moving_feat = self.feature(moving_smooth)

        fixed_low = F.avg_pool2d(fixed_feat, self.coarse_scale, stride=self.coarse_scale)
        moving_low = F.avg_pool2d(moving_feat, self.coarse_scale, stride=self.coarse_scale)
        corr_low = self._local_correlation(fixed_low, moving_low)
        coarse_input = torch.cat([fixed_low, moving_low, (fixed_low - moving_low).abs(), corr_low], dim=1)
        coarse_flow_low = torch.tanh(self.coarse_head(coarse_input)) * (self.max_coarse_flow / self.coarse_scale)
        coarse_flow = F.interpolate(coarse_flow_low, size=fixed.shape[-2:], mode="bilinear", align_corners=True)
        coarse_flow = coarse_flow * self.coarse_scale

        moving_feat_coarse = self._warp(moving_feat, coarse_flow)
        corr_fine = self._local_correlation(fixed_feat, moving_feat_coarse)
        moving_smooth_coarse = self._warp(moving_smooth, coarse_flow)
        residual_map = (fixed_smooth - moving_smooth_coarse).abs().mean(dim=1, keepdim=True)
        residual_norm = self._normalize_map(residual_map)
        residual_reliability = torch.exp(-self.residual_conf_temperature * residual_norm).clamp(0.0, 1.0)
        scatter_score = None
        if self.region_mode == "sar_scatter":
            scatter_score = self._sar_scatter_score(fixed_smooth)
            gate_score = scatter_score
            threshold = self.scatter_threshold
        else:
            gate_score = corr_fine
            threshold = self.high_corr_threshold

        threshold = self._adaptive_threshold(gate_score, threshold)
        soft_gate = torch.sigmoid((gate_score - threshold) * self.gate_temperature)
        hard_gate = (gate_score >= threshold).to(gate_score.dtype)
        high_gate = hard_gate + soft_gate - soft_gate.detach()
        if self.residual_suppress > 0.0:
            suppress = 1.0 - self.residual_suppress * (1.0 - residual_reliability)
            high_gate = (high_gate * suppress).clamp(0.0, 1.0)

        if self.use_coarse_fine_split:
            fine_input = torch.cat(
                [fixed_feat, moving_feat_coarse, (fixed_feat - moving_feat_coarse).abs(), corr_fine, high_gate],
                dim=1,
            )
            fine_flow = torch.tanh(self.fine_head(fine_input)) * self.max_fine_flow
        else:
            fine_flow = torch.zeros_like(coarse_flow)
        low_gate = 1.0 - high_gate
        if self.use_coarse_fine_split and self.low_conf_flow_scale < 1.0:
            coarse_scale = high_gate + low_gate * self.low_conf_flow_scale
            coarse_flow = coarse_flow * coarse_scale
        flow = coarse_flow + high_gate * fine_flow if self.use_coarse_fine_split else coarse_flow
        aligned_raw = self._warp(moving, flow)
        blend_source = scatter_score if scatter_score is not None else corr_fine
        blend = self._confidence_blend(blend_source)
        if self.residual_suppress > 0.0:
            blend = blend * (1.0 - self.residual_suppress * (1.0 - residual_reliability))
        if self.change_preserve_strength > 0.0:
            change_like = torch.sigmoid(
                (residual_norm - self.change_preserve_threshold) * self.change_preserve_temperature
            )
            blend = blend * (1.0 - self.change_preserve_strength * change_like)
        aligned = blend * aligned_raw + (1.0 - blend) * moving

        aux = {
            "fixed_feat": fixed_feat,
            "moving_feat": moving_feat,
            "corr_low": corr_low,
            "corr_fine": corr_fine,
            "high_gate": high_gate,
            "low_gate": low_gate,
            "blend": blend,
            "residual_reliability": residual_reliability,
            "residual_norm": residual_norm,
            "aligned_raw": aligned_raw,
            "coarse_flow": coarse_flow,
            "fine_flow": fine_flow,
        }
        if self.change_preserve_strength > 0.0:
            aux["change_like"] = change_like
        if scatter_score is not None:
            aux["scatter_score"] = scatter_score
        return flow, aligned, aux

    def _filter_speckle(self, x):
        return self.speckle_filter(x) if self.use_speckle_filter else x

    def _local_correlation(self, x1, x2):
        kernel = self.corr_window
        padding = kernel // 2
        mean1 = F.avg_pool2d(x1, kernel, stride=1, padding=padding)
        mean2 = F.avg_pool2d(x2, kernel, stride=1, padding=padding)
        z1 = x1 - mean1
        z2 = x2 - mean2
        cov = F.avg_pool2d(z1 * z2, kernel, stride=1, padding=padding).mean(dim=1, keepdim=True)
        var1 = F.avg_pool2d(z1 * z1, kernel, stride=1, padding=padding).mean(dim=1, keepdim=True)
        var2 = F.avg_pool2d(z2 * z2, kernel, stride=1, padding=padding).mean(dim=1, keepdim=True)
        corr = cov / (torch.sqrt(var1 * var2 + self.eps))
        return corr.clamp(-1.0, 1.0)

    def _warp(self, x, flow):
        b, _, h, w = x.shape
        y, x_coord = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        base_grid = torch.stack((x_coord, y), dim=-1).unsqueeze(0).expand(b, h, w, 2)
        norm_flow = torch.empty_like(flow)
        norm_flow[:, 0] = flow[:, 0] * (2.0 / max(w - 1, 1))
        norm_flow[:, 1] = flow[:, 1] * (2.0 / max(h - 1, 1))
        grid = base_grid + norm_flow.permute(0, 2, 3, 1)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def _edge_aware_smoothness(self, flow, image, mask=None):
        flow_dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs()
        flow_dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs()
        img_dx = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
        img_dy = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
        weight_x = torch.exp(-img_dx)
        weight_y = torch.exp(-img_dy)
        if mask is None:
            return (flow_dx * weight_x).mean() + (flow_dy * weight_y).mean()
        mask_x = mask[:, :, :, 1:]
        mask_y = mask[:, :, 1:, :]
        smooth_x = (flow_dx * weight_x * mask_x).sum() / mask_x.sum().clamp_min(1.0)
        smooth_y = (flow_dy * weight_y * mask_y).sum() / mask_y.sum().clamp_min(1.0)
        return smooth_x + smooth_y

    def _masked_l1(self, x, y, mask=None):
        if mask is None:
            return (x - y).abs().mean()
        denom = mask.sum().clamp_min(1.0)
        if mask.size(1) == 1 and x.size(1) > 1:
            denom = denom * x.size(1)
        return ((x - y).abs() * mask).sum() / denom

    def _prepare_mask(self, mask, reference):
        if mask is None:
            return None
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(device=reference.device, dtype=reference.dtype)
        if mask.shape[-2:] != reference.shape[-2:]:
            mask = F.interpolate(mask, size=reference.shape[-2:], mode="nearest")
        return mask.clamp(0.0, 1.0)

    def _confidence_blend(self, corr):
        denom = max(1.0 - self.blend_corr_start, self.eps)
        confidence = ((corr - self.blend_corr_start) / denom).clamp(0.0, 1.0)
        confidence = confidence * confidence
        return self.blend_floor + (1.0 - self.blend_floor) * confidence

    def _adaptive_threshold(self, score, base_threshold):
        if isinstance(base_threshold, torch.Tensor):
            threshold = base_threshold.to(device=score.device, dtype=score.dtype)
        else:
            threshold = score.new_full((score.size(0), 1, 1, 1), float(base_threshold))

        if self.adaptive_gate_alpha <= 0.0:
            return threshold

        target_ratio = float(self.target_high_ratio)
        target_ratio = max(min(target_ratio, 0.99), 0.01)
        flat = score.flatten(2)
        adaptive = torch.quantile(flat, q=1.0 - target_ratio, dim=-1).view(score.size(0), 1, 1, 1)
        alpha = float(self.adaptive_gate_alpha)
        threshold = (1.0 - alpha) * threshold + alpha * adaptive

        min_ratio = float(self.min_high_ratio)
        max_ratio = float(self.max_high_ratio)
        min_ratio = max(min(min_ratio, 0.99), 0.0)
        max_ratio = max(min(max_ratio, 0.99), min_ratio + 1e-3)
        thr_low = torch.quantile(flat, q=1.0 - max_ratio, dim=-1).view(score.size(0), 1, 1, 1)
        thr_high = torch.quantile(flat, q=1.0 - min_ratio, dim=-1).view(score.size(0), 1, 1, 1)
        return torch.minimum(torch.maximum(threshold, thr_low), thr_high)

    def _sar_scatter_score(self, sar):
        intensity = sar.mean(dim=1, keepdim=True)
        intensity = ((intensity + 1.0) * 0.5).clamp(0.0, 1.0)
        local_mean = F.avg_pool2d(intensity, kernel_size=7, stride=1, padding=3)
        local_sq = F.avg_pool2d(intensity * intensity, kernel_size=7, stride=1, padding=3)
        local_std = (local_sq - local_mean * local_mean).clamp_min(0.0).sqrt()

        grad_x = (intensity[:, :, :, 1:] - intensity[:, :, :, :-1]).abs()
        grad_y = (intensity[:, :, 1:, :] - intensity[:, :, :-1, :]).abs()
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        gradient = grad_x + grad_y

        contrast = local_std / (local_mean + self.eps)
        score = 0.55 * intensity + 0.25 * self._normalize_map(gradient) + 0.20 * self._normalize_map(contrast)
        return score.clamp(0.0, 1.0)

    def _normalize_map(self, x):
        flat = x.flatten(2)
        min_v = flat.min(dim=-1)[0].view(x.size(0), x.size(1), 1, 1)
        max_v = flat.max(dim=-1)[0].view(x.size(0), x.size(1), 1, 1)
        return (x - min_v) / (max_v - min_v + self.eps)
