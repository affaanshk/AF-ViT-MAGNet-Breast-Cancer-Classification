# Documentation & Architecture Visualizations

This directory contains the core architectural schematics, multi-model benchmarking pipelines, loss convergence curves, and comparative evaluation charts for **AF-ViT-MAGNet** and the evaluated baseline networks.

---

## 🖼️ Included Visual Assets

### 1. `architecture.png`
* **Description:** Detailed structural schematic of the proposed **AF-ViT-MAGNet** (Adaptive-Fusion Vision Transformer with Multi-Attention-Gating Network) architecture.
* **Key Components:**
  * Patch Embedding module ($16 \times 16$ spatial patches).
  * Vision Transformer (ViT) Encoder with Multi-Head Self-Attention and MLP layers.
  * **Mask-Attention Gate (MAG):** Novel spatial gating module utilizing segmentation priors to focus feature extraction on diagnostically relevant breast tissue.
  * Local + Global Attention Feature Fusion and Softmax Classification Head.

---

### 2. `benchmark_pipeline.png`
* **Description:** End-to-end multi-model benchmarking and evaluation workflow.
* **Workflow Stages:**
  1. Input of standardized, preprocessed screening mammograms.
  2. Patient-wise stratified splitting into Train, Validation, and Test subsets.
  3. Controlled comparative training across 9 model architectures (CNNs, Vision Transformers, Hybrids, and MLP-Mixer).
  4. Final evaluation tracking Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.

---

### 3. `loss_curves.png`
* **Description:** Comparative training and validation loss convergence curves across all 9 benchmarked architectures over training epochs.
* **Evaluated Models:** CNN, ResNet50, EfficientNet-B3, ConvNeXt-Tiny, Swin-Tiny, DeiT-Tiny, MLP-Mixer, ViT-B/16, and AF-ViT-MAGNet (Proposed).
* **Observation:** Highlights optimization stability, convergence speed, and generalization capability without overfitting under the AdamW optimization schedule.

---

### 4. `performance_chart.png`
* **Description:** Consolidated performance bar chart comparing all evaluated classification models on the Mini-DDSM test benchmark.
* **Metrics Reported:** Accuracy (%), Precision (%), Recall (%), and F1 Score (%) across BI-RADS-aligned risk categories.
