# Multi-Modal Deep Learning for NSCLC Prognosis

This repository contains the official implementation of the **Multi-Modal Deep Learning (MMDL)** pipeline for predicting Overall Survival (OS) in Non-Small Cell Lung Cancer (NSCLC) patients.

The framework integrates three distinct data modalities to improve prognostic accuracy:
1.  **Whole Slide Images (WSI):** Pathology features processed via ResNet50 + MIL Attention (encoded as 512-dim vectors).
2.  **Next-Generation Sequencing (NGS):** Somatic mutations, Copy Number Alterations (CNA), and Gene Sets processed via specialized MLPs and SNNs (Self-Normalizing Networks).
3.  **Clinical Data:** Patient demographics and status processed via MLP.

## ⚙️ System Requirements

* **OS:** Linux (Tested on Ubuntu 20.04/22.04)
* **GPU:** NVIDIA GPU with CUDA support (Tested on **NVIDIA RTX 4090**)
* **Python:** 3.10

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yangxu21/NSCLC_multimodel_deep_learning.git](https://github.com/yangxu21/NSCLC_multimodel_deep_learning.git)
    cd NSCLC_multimodel_deep_learning
    ```

2.  **Create the Conda Environment:**
    We provide a pre-configured environment file optimized for stable PyTorch/CUDA 12.1.
    ```bash
    conda env create -f config/environment.yml
    conda activate msclc_dl
    ```

## 📂 Data Preparation

The pipeline requires two main data sources. Please ensure your data matches the formats below before running the training scripts.

### 1. Tabular Data (CSV)
Prepare `mmdl_train.csv` and `mmdl_val.csv` containing **all** features concatenated. The scripts expect specific column indices:

| Index Range | Feature Type | Description |
| :--- | :--- | :--- |
| **0 - 3** | Metadata | `slide_id`, `survival_months`, `dead`, etc. |
| **4 - 104** | Mutation Data | 100 binary features (Gene level) |
| **104 - 204** | Mutation Gene Sets | 100 binary features (Pathway level) |
| **204 - 304** | CNA Amplification | 100 continuous/binary features |
| **304 - 404** | CNA Deletion | 100 continuous/binary features |
| **404 - 504** | Amp Gene Sets | 100 features |
| **504 - 604** | Del Gene Sets | 100 features |
| **604 - 609** | Clinical Data | 5 clinical features (Age, Stage, etc.) |

### 2. WSI Features (Patient Vectors)
The fusion model expects pre-extracted **512-dimensional** feature vectors for each patient (slide). Save these as PyTorch tensors (`.pt`) in a directory:
```text
processed_data/wsi_patient_vectors_512/
├── SLIDE_001.pt   # Shape: torch.Size([512])
├── SLIDE_002.pt
└── ...