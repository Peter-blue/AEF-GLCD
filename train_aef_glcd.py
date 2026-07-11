from __future__ import print_function

import os
import runpy
import sys


DEFAULTS = [
    ("--use_pretrained", "True"),
    ("--use_cgdr", "True"),
    ("--lambda_cgdr", "0.02"),
    ("--opt_cgdr_region_mode", "correlation"),
    ("--sar_cgdr_region_mode", "sar_scatter"),
    ("--cgdr_use_speckle_filter", "True"),
    ("--cgdr_use_coarse_fine_split", "True"),
    ("--lee_filter_sar_cd", "True"),
    ("--sar_lee_filter_mode", "auto"),
    ("--opt_lee_filter_mode", "none"),
    ("--use_sabr", "False"),
    ("--use_gbcc", "False"),
    ("--use_ucef", "False"),
    ("--use_racr", "False"),
    ("--use_barc", "False"),
    ("--checkpoint_interval", "-1"),
    ("--save_generated_images", "False"),
    ("--model_select_metric", "mean_f1"),
]


def append_defaults(defaults):
    user_keys = {arg.split("=", 1)[0] for arg in sys.argv[1:] if arg.startswith("--")}
    for key, value in defaults:
        if key not in user_keys:
            sys.argv.extend([key, value])


if __name__ == "__main__":
    append_defaults(DEFAULTS)
    train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
    runpy.run_path(train_path, run_name="__main__")
