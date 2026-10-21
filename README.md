# BRAHMA_CV50
### ବ୍ରହ୍ମା_ସିଭି୫୦ | ब्रह्मा_सीवी50

> **Biphasic Residual Architecture for High-Accuracy Multiclass Assessment (Computer Vision 50-Layer)**  
> Engineered by **Namish Kumar** ([@Learnr4Namish](https://github.com/Learnr4Namish)) in **Balasore, Odisha, India**.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Device-Apple%20Silicon%20MPS-black.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

**BRAHMA_CV50** is a fine-tuned 50-layer deep convolutional neural network specialized for fine-grained fruit and vegetable classification across 36 natural produce categories. 

Built on top of a pre-trained ResNet-50 backbone, the model uses a custom dense classification head (Dropout, Linear, LayerNorm, and ReLU) alongside a disciplined two-stage transfer learning schedule. It is packaged with an asynchronous FastAPI inference service and a clean frontend interface.

---

## 🏗️ Architecture Specification

The network replaces the standard 1,000-class linear projection head of ResNet-50 with a multi-layer regularized classifier head designed to prevent overfitting on subtle intra-class variances:

```text
Input Image (3 x 224 x 224)
         │
[ ResNet-50 Feature Extractor ]
├── Conv1 -> MaxPool -> Layer 1 (Conv2_x)
├── Layer 2 (Conv3_x)
├── Layer 3 (Conv4_x)
└── Layer 4 (Conv5_x) ───────────────> Bottleneck (2048 x 7 x 7)
         │
[ AdaptiveAvgPool2d ] ───────────────> Spatial Reduction (2048 x 1 x 1)
         │
[ Flatten ] ─────────────────────────> Feature Vector (2048)
         │
[ Classifier Head ]
├── Dropout (p = 0.4)
├── Linear (2048 -> 512)
├── LayerNorm (512)
├── ReLU (inplace = True)
├── Dropout (p = 0.2)
└── Linear (512 -> 36 classes) ──────> Logits Output (36)