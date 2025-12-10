# RADEMIC: Realistic Augmentation and Deep Learning for Enhanced Medical Image Classification of Alzheimer's Disease

A comprehensive deep learning approach for Alzheimer's disease classification from MRI scans, addressing critical limitations in existing research through methodological improvements. This project implements multiple architectures (CNN, CNN-LSTM, CNN-SVM, VGG16-SVM) with medically-consistent augmentation strategies.

## 👥 Team Members

- **Abdullah** - Project Lead / Main Contributor
  - Email: I221879@nu.edu.pk
  - Affiliation: School of Computing, FAST NUCES, Islamabad, Pakistan

- **Abdur Rahman Grami** - Contributor
  - Email: I222008@nu.edu.pk
  - Affiliation: School of Computing, FAST NUCES, Islamabad, Pakistan

- **Yousha Saibi** - Contributor
  - Email: L227482@isb.nu.edu.pk
  - Affiliation: School of Computing, FAST NUCES, Islamabad, Pakistan

- **Dr. Qurat Ul Ain** - Supervisor
  - Email: quratul.ain@isb.nu.edu.pk
  - Affiliation: School of Computing, FAST NUCES, Islamabad, Pakistan

## 📋 Project Overview

This project implements and improves upon existing Alzheimer's disease classification research by identifying and addressing critical issues including data leakage, unrealistic augmentation strategies, and class imbalance. We propose a medically-consistent methodology that achieves reliable and interpretable results.

## 🎯 Key Contributions (Novelties)

### 1. **Data Leakage Correction** (Major Novelty)
- **Problem Identified**: Original dataset had 100% data leakage (train and test sets were identical)
- **Solution**: Created a clean, properly stratified dataset with zero overlap
- **Impact**: Exposed inflated accuracy metrics (99.92% → realistic ~85%)
- **Files**: `Improvements_or_Novelty/01_fix_dataset.ipynb`

### 2. **Clean Dataset Creation**
- **Contribution**: Created `Alzheimer_Clean_Dataset/` with proper 80/20 train/test split
- **Features**: 
  - 5,120 training images (80%)
  - 1,280 test images (20%)
  - Balanced classes with no duplicates
- **Impact**: Enables reliable benchmarking for future research

### 3. **Realistic Augmentation Strategy**
- **Problem**: Base paper used aggressive augmentation (90° rotation + flips) creating anatomically unrealistic images
- **Solution**: Conservative, medically-consistent augmentation:
  - ±15° rotation only (natural head tilt range)
  - No horizontal/vertical flips (preserves brain anatomy)
  - Dynamic augmentation (new transforms each epoch)
- **Impact**: Improved accuracy from 60-66% (static aggressive) → 85%+ (realistic dynamic)
- **Files**: `Improvements_or_Novelty/Step1_Realistic_Augmentation.ipynb`

### 4. **Class Imbalance Correction**
- **Problem**: Severe class imbalance (ModerateDemented: 1% of samples)
- **Solution**: Systematic comparison of two techniques:
  - Class-weighted loss
  - Focal Loss
- **Impact**: Improved recall on minority class while maintaining overall accuracy
- **Files**: `Improvements_or_Novelty/Step3_Class_Imbalance_Correction.ipynb`

### 5. **Grad-CAM Explainability**
- **Contribution**: Visual explanations showing which brain regions the model focuses on
- **Impact**: 
  - Enhanced clinical interpretability
  - Validates model focuses on diagnostically relevant regions (hippocampus, ventricles)
  - Critical for FDA approval and clinical trust
- **Files**: `Improvements_or_Novelty/Step2_GradCAM_Explainability.ipynb`

## 📁 Repository Structure

```
Classification-of-Alzheimer-s-disease-MRI-data-using-Deep-Learning/
│
├── Improvements_or_Novelty/          # Main improvements and novelties
│   ├── improved-methodology.ipynb   # Complete pipeline: CNN with realistic aug + Grad-CAM + class imbalance
│   ├── CNN-LSTM-with-augmentation.ipynb  # Standalone CNN-LSTM model (Kaggle-ready)
│   ├── class-imbalance.ipynb        # Class imbalance correction experiments
│   ├── conference_101719.tex        # IEEE conference paper (LaTeX)
│   └── 01_fix_dataset.ipynb         # Dataset cleaning and leakage correction
│
├── Replication of base paper/         # Base paper replication
│   ├── 01_fix_dataset.ipynb
│   ├── 02_preprocessing_clean.ipynb
│   ├── 03-train-models.ipynb
│   └── preprocessed_data/
│
├── Results/                           # Experimental results and outputs
│   ├── CNN-LSTM_Realistic_Augmentation_Results.csv
│   ├── Step1_Realistic_Augmentation_Results.csv
│   ├── Step3_Class_Imbalance_Results.csv
│   ├── CNN_Realistic_Aug_model.h5
│   └── IEEE_*.png                    # IEEE-standard visualizations
│
├── Alzheimer_Clean_Dataset/          # Clean dataset (no leakage)
│   ├── train/                        # 5,120 images (80%)
│   │   ├── NonDemented/              # 2,560 images
│   │   ├── VeryMildDemented/         # Mapped to "Demented"
│   │   ├── MildDemented/             # Mapped to "Demented"
│   │   └── ModerateDemented/         # Mapped to "Demented"
│   └── test/                         # 1,280 images (20%)
│       └── [same structure]
│
└── README.md                          # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow>=2.18.0 keras numpy pandas matplotlib seaborn scikit-learn opencv-python joblib
```

### Running Models on Kaggle (Recommended)

All models have standalone Kaggle-ready notebooks in `Improvements_or_Novelty/`:

1. **CNN with Realistic Augmentation**:
   - Notebook: `improved-methodology.ipynb`
   - Dataset: Add `alzheimer-clean-dataset` to Kaggle
   - Enable GPU accelerator
   - Update `DATA_ROOT` if needed (default: `/kaggle/input/alzheimer-clean-dataset/Alzheimer_Clean_Dataset`)

2. **CNN-LSTM with Augmentation**:
   - Notebook: `CNN-LSTM-with-augmentation.ipynb`
   - Same dataset setup as above
   - Batch size: 8 (optimized for memory)

3. **CNN-SVM with Augmentation**:
   - Notebook: `CNN_SVM_with_augmentation.ipynb` (if available)
   - Batch size: 12 for feature extraction

4. **VGG16-SVM with Augmentation**:
   - Notebook: `VGG16_SVM_with_augmentation.ipynb` (if available)
   - Batch size: 6 (VGG16 requires more memory)
   - Input size: 224×224

### Running Locally

1. **Update DATA_ROOT** in each notebook:
   ```python
   DATA_ROOT = r"D:\...\Alzheimer_Clean_Dataset"
   ```

2. **Run notebooks sequentially**:
   - Start with `improved-methodology.ipynb` for CNN baseline
   - Then run individual model notebooks for comparisons

## 📊 Results Summary

### Model Performance Comparison

| Model | Augmentation | Imbalance Correction | Accuracy | Recall | F1-Score | Precision |
|-------|-------------|---------------------|----------|--------|----------|-----------|
| Baseline (No Aug) | None | None | 98.91% | 0.9891 | 0.9891 | 0.9891 |
| Aggressive Aug | 90° rot., flips | None | 60-66% | 0.60-0.66 | 0.61-0.67 | 0.62-0.68 |
| Realistic Aug | ±15°, no flips | None | 85%+ | 0.85+ | 0.85+ | 0.85+ |
| **CNN (Baseline)** | Realistic | None | **97.89%** | **0.9719** | **0.9788** | **0.9857** |
| **CNN (Class Weights)** | Realistic | Class-weighted | **98.28%** | **0.9797** | **0.9828** | **0.9858** |
| CNN (Focal Loss) | Realistic | Focal loss | 97.97% | 0.9750 | 0.9796 | 0.9842 |
| **CNN-LSTM** | Realistic | None | **92.34%** | **0.9063** | **0.9221** | **0.9385** |

### Key Findings

- **Best Performance**: CNN with class-weighted loss achieves **98.28% accuracy** and **97.97% recall**
- **Data Leakage Impact**: Corrected dataset reveals realistic performance (97.89%) vs. inflated metrics (99.92%)
- **Augmentation Strategy**: Realistic augmentation outperforms aggressive augmentation by **22-25%**
- **Class Imbalance Correction**: Class-weighted loss improves recall by **+0.78%** (critical for minority class detection)
- **CNN-LSTM**: Hybrid architecture achieves **92.34% accuracy** with **94.06% specificity**
- **Explainability**: Grad-CAM validates model focuses on clinically relevant brain regions (hippocampus, ventricles)

## 🔬 Methodology

### Model Architectures

#### 1. CNN Architecture (Primary)
- **Input Size**: 128×128×3 (RGB)
- **Convolutional Blocks**: 
  - Block 1: Two 32-filter 3×3 convs + BN + MaxPool + 25% dropout
  - Block 2: Two 64-filter 3×3 convs + BN + MaxPool + 25% dropout
  - Block 3: Two 128-filter 3×3 convs + BN + MaxPool + 30% dropout
  - Block 4: One 256-filter 3×3 conv + BN + MaxPool + 30% dropout
- **Dense Layers**: 512→256→128→2 with BN and dropout (40-50%)
- **Total Parameters**: ~2.5 million
- **Best Performance**: 98.28% accuracy with class-weighted loss

#### 2. CNN-LSTM Hybrid Architecture
- **CNN Feature Extraction**: Same 4 blocks as CNN (32→64→128→256 filters)
- **Reshape**: 8×8×256 → 64×256 sequence format
- **Time-Distributed Dense**: 128 units with BN and 35% dropout
- **LSTM Layers**: Single LSTM (64 units) with dropout (0.4) and recurrent dropout (0.4)
- **Classification Head**: 128→64→2 with BN and dropout (50%)
- **Regularization**: L2 regularization (1e-4 for conv/LSTM, 1e-3 for dense)
- **Total Parameters**: ~1.8 million
- **Performance**: 92.34% accuracy, 90.63% recall, 94.06% specificity

### Training Configuration

- **Optimizer**: Adam (learning_rate=0.001, beta_1=0.9, beta_2=0.999)
- **Loss Function**: 
  - Baseline: Sparse Categorical Crossentropy
  - Class-weighted: Inverse class frequency weighting
  - Focal Loss: γ=2.0, α=0.25
- **Batch Size**: 
  - CNN: 64 (optimized for GPU)
  - CNN-LSTM: 8 (memory-efficient)
- **Epochs**: 150 with early stopping (patience=20)
- **Callbacks**: 
  - EarlyStopping (monitor='val_accuracy', patience=20)
  - ReduceLROnPlateau (factor=0.5, patience=10, min_lr=1e-6)

### Augmentation Strategy

**Realistic Augmentation** (Our Method):
- Rotation: ±15° (natural head tilt)
- Flips: None (preserves anatomy)
- Zoom: ±10%
- Shift: ±10%
- Brightness: ±10%

**Aggressive Augmentation** (Base Paper):
- Rotation: 0-90° ❌
- Flips: Horizontal + Vertical ❌
- Zoom: ±15%
- Shift: ±15%

## 📈 Experimental Results

### CNN Model Results

**Baseline (Realistic Augmentation)**:
- Accuracy: 97.89%
- Recall: 97.19%
- F1-Score: 97.88%
- Precision: 98.57%

**Class-Weighted Loss (Best Model)**:
- Accuracy: **98.28%** (+0.39%)
- Recall: **97.97%** (+0.78%)
- F1-Score: **98.28%** (+0.40%)
- Precision: 98.58% (+0.01%)

**Focal Loss**:
- Accuracy: 97.97% (+0.08%)
- Recall: 97.50% (+0.31%)
- F1-Score: 97.96% (+0.08%)
- Precision: 98.42% (-0.15%)

### CNN-LSTM Model Results

- Accuracy: **92.34%**
- Recall: 90.63%
- F1-Score: 92.21%
- Precision: 93.85%
- Specificity: 94.06%

### Key Improvements

1. **Realistic Augmentation**: +22-25% improvement over aggressive augmentation (60-66% → 85%+)
2. **Class Imbalance Correction**: +0.78% recall improvement (critical for minority class detection)
3. **Data Leakage Correction**: Realistic metrics (97.89%) vs. inflated (99.92%)
4. **Grad-CAM Validation**: Model focuses on clinically relevant regions (hippocampus, ventricles)

## 📝 Paper Information

### Conference Paper

**Title**: RADEMIC: Realistic Augmentation and Deep Learning for Enhanced Medical Image Classification of Alzheimer's Disease

**Format**: IEEE Conference Paper (LaTeX)

**Location**: `Improvements_or_Novelty/conference_101719.tex`

### Abstract Summary

> "Alzheimer's disease (AD) is a progressive neurodegenerative disorder affecting millions worldwide. Early and accurate diagnosis using magnetic resonance imaging (MRI) is crucial for effective treatment planning. This paper addresses critical limitations in existing deep learning approaches for AD classification: data leakage between training and testing sets, anatomically unrealistic data augmentation strategies, and severe class imbalance. We propose three key contributions: (1) a realistic augmentation pipeline that preserves anatomical consistency, (2) gradient-weighted class activation mapping (Grad-CAM) for model explainability, and (3) systematic evaluation of class imbalance correction techniques including class-weighted loss and focal loss. Using a convolutional neural network (CNN) architecture, our method achieves 98.28% accuracy with 97.97% recall on a properly stratified dataset of 6,400 MRI images."

### Key Sections

1. **Introduction**: Problem statement and limitations of existing work
2. **Related Work**: Review of base paper and data leakage issues
3. **Methodology**: 
   - Dataset description and binary label mapping
   - Model architectures (CNN, CNN-LSTM)
   - Realistic augmentation strategy
   - Grad-CAM for explainability
   - Class imbalance correction techniques
   - Training configuration
4. **Experimental Results**: 
   - Comprehensive performance comparison (Table I)
   - Class imbalance correction results (Table II)
   - CNN-LSTM performance analysis
   - Grad-CAM visualizations
   - Quantitative improvements summary
5. **Discussion**: Clinical implications and methodology importance
6. **Conclusion**: Future work and clinical deployment considerations

## 🛠️ Technical Details

### Dataset

- **Source**: Alzheimer's Disease Neuroimaging Initiative (ADNI)
- **Classes**: 4 original classes → 2 binary classes
  - NonDemented
  - Demented (VeryMild + Mild + Moderate)
- **Split**: 80% train / 20% test (stratified)
- **Total Images**: 6,400 unique images (no duplicates)

### Dependencies

```
tensorflow>=2.18.0
keras>=3.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
opencv-python>=4.8.0
```

### Hardware Requirements

- **Recommended**: GPU (NVIDIA with CUDA support)
- **Minimum**: CPU (slower training)
- **Memory**: 8GB+ RAM recommended

## 📄 Citation

If you use this work, please cite:

```bibtex
@inproceedings{rademic2025,
  title={RADEMIC: Realistic Augmentation and Deep Learning for Enhanced Medical Image Classification of Alzheimer's Disease},
  author={Abdullah and Grami, Abdur Rahman and Saibi, Yousha and Ain, Qurat Ul},
  booktitle={IEEE Conference Proceedings},
  year={2025},
  organization={IEEE}
}
```

## 📊 Output Files

All results and visualizations are saved in the `Results/` directory:

- **CSV Results**: Performance metrics for all models
- **Model Files**: Trained models (`.h5` format)
- **IEEE Visualizations**: High-resolution figures for paper (300 DPI, Times New Roman)
  - Confusion matrices
  - ROC curves
  - Training curves
  - Grad-CAM heatmaps
  - Class distribution charts
  - Metrics comparisons

## 📧 Contact

For questions or collaborations, please open an issue or contact the repository maintainer.

## 📜 License

See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Base paper authors for the initial research
- ADNI dataset contributors
- Open-source deep learning community

---

**Note**: This repository focuses on **methodological improvements** rather than architecture diversity. The novelties address critical issues in medical AI research: data quality, training methodology, and interpretability.
