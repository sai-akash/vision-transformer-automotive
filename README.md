# Vision Transformer Automotive 

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/docs/transformers/index)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Implemented **Vision Transformers (ViT)** for automotive image classification (traffic signs & car models).  
> Extended to **object detection** with DETR, including **attention visualizations** to highlight discriminative features.  

---

## Features
- **Classification**
  - Traffic sign recognition on **[GTSRB](https://benchmark.ini.rub.de/gtsrb_news.html)**  
  - Car model recognition on **[Stanford Cars](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)**  
- **Detection**
  - Object detection with **DETR** (COCO-style annotations)  
- **Interpretability**
  - **Attention rollout** heatmaps to visualize where the transformer is “looking”
    

