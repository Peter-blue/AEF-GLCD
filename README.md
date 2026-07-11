# AEF-GLCD

Official implementation skeleton of **AEF-GLCD** for heterogeneous optical/SAR remote-sensing change detection. The framework contains three stages:

1. **AEF / HuiYan embedding** translates heterogeneous observations into cross-modal auxiliary images.
2. **CGDR** performs correlation-guided deformable registration with SAR speckle suppression and coarse/fine alignment.
3. **SNUNet-ECAM** predicts bidirectional change maps in the optical and SAR domains.

This public package intentionally excludes datasets, pretrained weights, checkpoints, logs, and experimental outputs.

## Repository structure

```text
AEF-GLCD/
|-- Model/
|   |-- cgdr.py             # correlation-guided deformable registration
|   |-- Sun_Net.py          # SNUNet-ECAM change detector
|   `-- Sun_Net_gan.py      # registration/CD interface
|-- models/
|   `-- generator.py        # HuiYan generator adapter
|-- HuiYanEarth-SAR/
|   |-- model.py            # lightweight generator definition
|   `-- pretrained/         # user-provided pretrained weights
|-- loss/
|-- dataset/                # user-provided datasets
|-- checkpoint/             # best checkpoints produced by training
|-- result/                 # quantitative and visual results
|-- train_aef_glcd.py       # recommended AEF-GLCD training entry
|-- train.py                # complete configurable training script
|-- test.py                 # patch-level evaluation
|-- test_full_scene.py      # sliding-window full-scene evaluation
`-- test_artificial_misalignment.py
```

## Environment

Python 3.9 or later is recommended. Install PyTorch and torchvision for the CUDA version available on the machine, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Check the code path without downloading data or weights:

```bash
python smoke_test.py
```

## Data preparation

The expected directory structure is documented in `dataset/README.md`. For example:

```text
dataset/Gloucester/train/Image/
dataset/Gloucester/train/Image2/
dataset/Gloucester/train/label/
dataset/Gloucester/test/Image/
dataset/Gloucester/test/Image2/
dataset/Gloucester/test/label/
```

Images in `Image`, `Image2`, and `label` are matched by filename stem.

## Pretrained HuiYan generators

Place the pretrained translation weights at:

```text
HuiYanEarth-SAR/pretrained/huiyan_sar_v1.pth
HuiYanEarth-SAR/pretrained/huiyan_opt_v1.pth
```

The weights are omitted from GitHub because of their size and third-party distribution requirements.

## Training

The recommended entry enables HuiYan and the complete CGDR configuration while disabling unrelated experimental constraints:

```bash
python train_aef_glcd.py \
  --dataset Gloucester \
  --batch_size 2 \
  --test_batch_size 1 \
  --epochs 500 \
  --num_workers 4 \
  --pin_memory True \
  --persistent_workers True \
  --prefetch_factor 2 \
  --cuda \
  --gpu_id 0
```

Only the best model is retained by default. The four files are written to `checkpoint/<EXPERIMENT_NAME>/`:

```text
G_opt2sar_best.pth
G_sar2opt_best.pth
D_opt_best.pth
D_sar_best.pth
```

## Evaluation

Patch-level evaluation:

```bash
python test.py \
  --dataset Gloucester \
  --exp_name <EXPERIMENT_NAME> \
  --use_cgdr True \
  --cuda \
  --gpu_id 0
```

Full-scene sliding-window evaluation:

```bash
python test_full_scene.py \
  --scene California \
  --exp_name <EXPERIMENT_NAME> \
  --use_cgdr True \
  --patch_size 256 \
  --stride 128 \
  --cuda \
  --gpu_id 0
```

Evaluation reports OA, Precision, Recall, F1, and IoU for both translation directions and stores confusion maps under `result/`.

## Ablation studies

The following wrappers reproduce the principal module ablations:

```text
train_ablation_huiyan_only.py
train_ablation_pixel_alignment.py
train_ablation_full_cgdr.py
train_ablation_cgdr_no_sar_scatter.py
train_ablation_cgdr_no_lee_filter.py
train_ablation_cgdr_no_coarse_fine.py
```

They accept the same dataset, epoch, data-loader, and CUDA arguments as `train.py`.

## Artificial misalignment protocol

Use `test_artificial_misalignment.py` to evaluate robustness to controlled spatial offsets. An example using a shifted SAR input is:

```bash
python test_artificial_misalignment.py \
  --mode full_scene \
  --scene California \
  --exp_name <EXPERIMENT_NAME> \
  --use_cgdr True \
  --misalignment_stage input \
  --input_shift_modality sar \
  --valid_overlap_only True \
  --shift_list "0,0;2,0;4,0;8,0" \
  --cuda \
  --gpu_id 0
```

## Notes for public release

- Verify the redistribution terms of HuiYan, MTCDN, and SNUNet-derived components before adding a repository license.
- Add the final paper citation and pretrained-weight download links after publication.
- Do not commit datasets, generated images, or model checkpoints; these paths are covered by `.gitignore`.
