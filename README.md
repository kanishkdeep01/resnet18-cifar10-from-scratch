# Elevated ResNet-18: Custom PyTorch Architecture on CIFAR-10

A custom 18-layer Residual Network (ResNet-18) built from scratch in PyTorch, engineered with custom data augmentation pipelines, L2 regularization, and Cosine Annealing learning rate scheduling to achieve a certified **93.20% test accuracy** on CIFAR-10.

![Confusion Matrix](confusion_matrix.png)

---

## Key Results & Metrics

Evaluated across 10,000 held-out test images:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Airplane** | 0.9313 | 0.9350 | 0.9331 | 1000 |
| **Automobile** | 0.9688 | 0.9620 | 0.9654 | 1000 |
| **Bird** | 0.9322 | 0.8940 | 0.9127 | 1000 |
| **Cat** | 0.8682 | 0.8630 | 0.8656 | 1000 |
| **Deer** | 0.9300 | 0.9560 | 0.9428 | 1000 |
| **Dog** | 0.8822 | 0.8910 | 0.8866 | 1000 |
| **Frog** | 0.9488 | 0.9630 | 0.9558 | 1000 |
| **Horse** | 0.9751 | 0.9410 | 0.9578 | 1000 |
| **Ship** | 0.9439 | 0.9590 | 0.9514 | 1000 |
| **Truck** | 0.9409 | 0.9560 | 0.9484 | 1000 |
| **Overall Accuracy** | — | — | **0.9320** | **10000** |

---

## Architectural Highlights

* **Custom Residual Blocks:** Implemented custom `BasicBlock` modules featuring $3 \times 3$ convolutional layers, Batch Normalization, shortcut residual skip-connections, and explicit $1 \times 1$ projections for spatial downsampling.
* **Regularization & Optimization Pipeline:**
  * **Data Augmentations:** `RandomCrop(32, padding=4)` and `RandomHorizontalFlip()` applied strictly during training.
  * **Weight Decay:** L2 penalty (`weight_decay=1e-4`) with Adam optimizer to limit parameter memorization.
  * **Learning Rate Scheduling:** `CosineAnnealingLR` scheduler decaying smoothly down across 50 training epochs.
* **Inference Defenses (`app.py`):** Production-ready Gradio web app including mandatory input sanitization—automated $32 \times 32$ image resizing and exact CIFAR-10 channel normalization $(\mu, \sigma)$ to eliminate out-of-distribution runtime failures.

---

## Repository Structure

```text
├── dataset.py                  # Augmentations & CIFAR-10 DataLoaders
├── model_resnet.py             # ResNet-18 architecture implementation
├── train.py                    # 50-epoch training loop & checkpointing
├── evaluate.py                 # Per-class metrics & confusion matrix generator
├── app.py                      # Gradio web application with defense pipeline
├── elevated_resnet_cifar10.pth # Saved model weights
├── confusion_matrix.png        # Generated evaluation heatmap
└── requirements.txt            # Python dependencies
```

---

## Quickstart

### 1. Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/kanishkdeep01/resnet18-cifar10-from-scratch.git
cd resnet18-cifar10-from-scratch
pip install torch torchvision gradio scikit-learn seaborn matplotlib
```

### 2. Model Evaluation
Generate the per-class performance report and save `confusion_matrix.png`:
```bash
python evaluate.py
```

### 3. Launch Web Interface
Boot the local interactive Gradio interface:
```bash
python app.py
```