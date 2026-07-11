from __future__ import print_function

import os
import runpy
import sys


COMMON_DEFAULTS = [
    ("--use_pretrained", "True"),
    ("--use_sabr", "False"),
    ("--use_gbcc", "False"),
    ("--use_ucef", "False"),
    ("--use_racr", "False"),
    ("--use_barc", "False"),
    ("--checkpoint_interval", "-1"),
    ("--save_generated_images", "False"),
    ("--export_epoch_metrics", "False"),
    ("--plot_epoch_curves", "False"),
    ("--profile_complexity", "False"),
    ("--profile_only", "False"),
    ("--model_select_metric", "mean_f1"),
    ("--lee_filter_sar_cd", "True"),
    ("--sar_lee_filter_mode", "auto"),
    ("--opt_lee_filter_mode", "none"),
]

VARIANT_DEFAULTS = [
    ("--use_cgdr", "False"),
    ("--lambda_cgdr", "0.0"),
    ("--lambda_cgdr_opt", "0.0"),
    ("--lambda_cgdr_sar", "0.0"),
]


def append_defaults(defaults):
    user_keys = {arg.split("=", 1)[0] for arg in sys.argv[1:] if arg.startswith("--")}
    ordered_keys = []
    merged = {}
    for key, value in defaults:
        if key not in merged:
            ordered_keys.append(key)
        merged[key] = value
    for key in ordered_keys:
        if key not in user_keys:
            sys.argv.extend([key, merged[key]])


if __name__ == "__main__":
    append_defaults(COMMON_DEFAULTS + VARIANT_DEFAULTS)
    print("[Ablation] Baseline + HuiYan + Pixel Alignment")
    train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
    runpy.run_path(train_path, run_name="__main__")
