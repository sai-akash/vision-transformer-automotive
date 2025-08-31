# Vision Transformers for Automotive — Starter

This repo extends your **Assignment 2 (Transformers)** to real automotive CV tasks.

## What's inside
- `notebooks/01_Assignment2_Transformers_solution.ipynb` — your original notebook (copied).
- `notebooks/02_ViT_Automotive.ipynb` — fine‑tune ViT on **GTSRB** and optionally **Stanford Cars** + attention visualization (rollout).
- `notebooks/03_DETR_Automotive.ipynb` — skeleton for fine‑tuning **DETR** for detection using COCO‑style annotations.
- `src/viz/attention_rollout.py` — utility for attention rollout.

## Quickstart
1. Create a new Python 3.10+ environment.
2. `pip install -r requirements.txt`
3. Open `notebooks/02_ViT_Automotive.ipynb` and run all cells to train on **GTSRB**.
4. (Optional) Prepare a small COCO dataset for traffic‑sign or car detection and follow `03_DETR_Automotive.ipynb`.

## Datasets
- **GTSRB** downloads automatically via `torchvision.datasets.GTSRB`.
- **Stanford Cars** can also be downloaded via `torchvision.datasets.StanfordCars`.

## Resume bullet (suggested)
*Implemented Vision Transformers (ViT) for automotive image classification (GTSRB, Stanford Cars); extended to object detection with DETR and built attention visualizations highlighting discriminative vehicle features.*
