# Dataset

**3,073 hand images**, hand-annotated in Roboflow with the polygon tool. There is one class, `fingernail`. Labels use the YOLOv8 segmentation format, with one `.txt` file per image.

| Split | Images |
|---|---|
| train | 2,155 |
| valid | 611 |
| test | 307 |

## What's included

- `data.yaml`: the dataset config (paths, `nc: 1`, `names: ['fingernail']`)
- `train/labels/`, `valid/labels/`, `test/labels/`: polygon annotations

## What's not included, and why

The **images are not redistributed** in this repository:

- **1,921 images** are a subset of the [11K Hands dataset](https://sites.google.com/view/11khands) (M. Afifi, 2019). It is licensed for research use, so please download it from the official source.
- **1,152 images** came from Google Images and Pinterest. They are third-party copyrighted content.

## Rebuilding the image folders

1. Download 11K Hands from the official page.
2. Each label file has the same base name as its image, for example `Hand_0000002.txt` ↔ `Hand_0000002.jpg`. Put each matching image into `<split>/images/`, next to its `<split>/labels/` folder.
3. Train with `python src/train.py --size n`.

Labels for the web-sourced images are kept for completeness. To train on 11K Hands only, delete the label files that have no matching image.
