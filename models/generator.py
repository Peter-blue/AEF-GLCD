import importlib.util
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn


def _load_huiyan_generator_cls():
    model_path = Path(__file__).resolve().parents[1] / "HuiYanEarth-SAR" / "model.py"
    if not model_path.exists():
        raise FileNotFoundError(f"HuiYan generator definition not found: {model_path}")
    spec = importlib.util.spec_from_file_location("huiyan_sar_model", str(model_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if not hasattr(module, "HuiYanSARGenerator"):
        raise AttributeError(f"HuiYanEarth-SAR/model.py must define HuiYanSARGenerator, got: {model_path}")
    return module.HuiYanSARGenerator


HuiYanSARGenerator = _load_huiyan_generator_cls()


class GeneratorWrapper(nn.Module):
    """
    Thin wrapper so MTCDN can use HuiYanSARGenerator as drop-in generator.
    Input:  [B, 3, H, W]
    Output: [B, 3, H, W] in [-1, 1]
    """

    def __init__(self, input_nc: int = 3, output_nc: int = 3, n_residual_blocks: int = 9):
        super().__init__()
        self.net = HuiYanSARGenerator(input_nc=input_nc, output_nc=output_nc, n_residual_blocks=n_residual_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_generator(input_nc: int = 3, output_nc: int = 3, n_residual_blocks: int = 9) -> GeneratorWrapper:
    return GeneratorWrapper(input_nc=input_nc, output_nc=output_nc, n_residual_blocks=n_residual_blocks)


def _extract_state_dict(ckpt: Dict) -> Dict:
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "generator_state_dict", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError("Unsupported checkpoint format, expected state_dict-like dict.")


def load_pretrained_generator(model: nn.Module, weight_path: str, strict: bool = False) -> Tuple[list, list]:
    """
    Load pretrained weights to generator. Returns (missing_keys, unexpected_keys).
    """
    path = Path(weight_path)
    if not path.exists():
        raise FileNotFoundError(f"Pretrained weight not found: {path}")
    ckpt = torch.load(str(path), map_location="cpu")
    state_dict = _extract_state_dict(ckpt)
    result = model.load_state_dict(state_dict, strict=strict)
    return list(result.missing_keys), list(result.unexpected_keys)
