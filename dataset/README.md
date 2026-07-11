# Dataset layout

Datasets are intentionally excluded from this repository. Place each dataset under:

```text
dataset/<DATASET_NAME>/
|-- train/
|   |-- Image/   # optical image at time 1
|   |-- Image2/  # SAR or heterogeneous image at time 2
|   `-- label/   # binary change mask
`-- test/
    |-- Image/
    |-- Image2/
    `-- label/
```

Paired images and labels must share the same filename stem. Supported image formats are PNG, JPG, JPEG, TIF, and TIFF.
