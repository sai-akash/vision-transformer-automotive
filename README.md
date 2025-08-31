# Vision Transformer Automotive 

This project extends a university assignment on Transformers into **automotive computer vision** tasks.

## Features
- **Classification**
  - Traffic sign recognition (GTSRB)
  - Car model recognition (Stanford Cars, optional)
- **Detection**
  - Object detection with DETR (toy COCO & real datasets)
- **Interpretability**
  - Attention rollout visualizations highlighting discriminative features

## Project Structure
vision-transformer-automotive/
│
├── notebooks/
│   ├── 01_Assignment2_Transformers_solution.ipynb
│   ├── 02_ViT_Automotive.ipynb
│   └── 03_DETR_Automotive.ipynb
│
├── src/
│   └── viz/attention_rollout.py
├── requirements.txt
├── README.md
└── .gitignore

Quickstart:
1. Clone the repo
   git clone https://github.com/sai-akash/vision-transformer-automotive.git
   cd vision-transformer-automotive
2.Install dependencies
  pip install -r requirements.txt
3. Run notebooks
   02_ViT_Automotive.ipynb → classification + attention visualization
   03_DETR_Automotive.ipynb → object detection with DETR
