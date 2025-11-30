<div align="center">

# 🎵 Music Genre Classification

### *AI-Powered Audio Classification with Machine Learning*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

[📊 Demo](#-demo-scenarios) • [🚀 Quick Start](#-quick-start) • [📈 Results](#-results)

</div>

---

## 📋 Overview

A supervised machine learning system that classifies music tracks into **10 genres** using **Mel-Frequency Cepstral Coefficients (MFCCs)** as audio features. Implements and compares multiple classification algorithms with hyperparameter optimization.

### ✨ Key Features

- 🎯 **Multi-Model Comparison** - KNN, Decision Tree, Random Forest, SVM
- 🔧 **Hyperparameter Tuning** - GridSearchCV with cross-validation
- 📊 **Comprehensive Metrics** - Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- ⚡ **Fast Execution** - Optimized for 7-10 minute total runtime
- 📈 **Rich Visualizations** - Performance charts, confusion matrices, per-genre analysis

---

## 🎵 Dataset

**GTZAN Genre Collection**
- **Total Samples:** 1,000 audio tracks (100 per genre)
- **Genres:** blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- **Format:** WAV files
- **Duration:** 30 seconds per track
- **Sampling Rate:** 22,050 Hz

### Dataset Download

Download the GTZAN dataset from one of these sources:
1. [Kaggle - GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
2. [Marsyas GTZAN](http://marsyas.info/downloads/datasets.html)

After downloading, extract the dataset and update the `DATASET_PATH` variable in the notebook.

---

## 🏁 Quick Start

### 1️⃣ Clone & Navigate

```bash
git clone https://github.com/Papi-Hokage/music-genre-classification.git
cd music-genre-classification
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Download GTZAN Dataset

Download from [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) or [Marsyas](http://marsyas.info/downloads/datasets.html)

### 4️⃣ Run the Notebook

```bash
jupyter notebook "Music Genre Final Project.ipynb"
```

### 5️⃣ Execute Cells

Run cells sequentially - **Total runtime: ~7-10 minutes**

---

## 🎬 Demo Scenarios

### 1. Feature Extraction in Action

```python
# Extract MFCCs from audio
mfcc = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)
features = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
```

• Watch features being extracted from 1,000 audio tracks  
• Observe progress indicators for each 100 files processed  
• Verify feature matrix shape: `(1000, 26)`

### 2. Model Training & Comparison

```python
# Train multiple models
models = ['KNN', 'Decision Tree', 'Random Forest', 'SVM']
# Compare performance metrics
```

• See baseline vs tuned model performance  
• Visualize accuracy improvements after hyperparameter tuning  
• Identify best performing model

### 3. Confusion Matrix Analysis

• Discover which genres are confused with each other  
• Rock ↔ Country (similar instrumentation)  
• Classical ↔ Jazz (harmonic structures)  
• View detailed misclassification patterns

---

## 🎵 Dataset

**GTZAN Genre Collection**

| Property | Value |
|----------|-------|
| **Total Tracks** | 1,000 (100 per genre) |
| **Genres** | blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock |
| **Format** | WAV files |
| **Duration** | 30 seconds per track |
| **Sampling Rate** | 22,050 Hz |
| **Balance** | Perfectly balanced (equal samples per class) |

---

## 🏗️ Architecture Overview

### Pipeline Flow

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Audio     │ ─► │    MFCC      │ ─► │  Features   │
│   Files     │    │  Extraction  │    │ (Mean+Std)  │
│  (1000)     │    │  (13 coef)   │    │  (26-dim)   │
└─────────────┘    └──────────────┘    └─────────────┘
                            │
                            ▼
              ┌──────────────────────┐
              │   Train/Test Split   │
              │      (80/20)         │
              └──────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      ┌──────────────┐          ┌──────────────┐
      │   Training   │          │   Testing    │
      │   (4 Models) │          │ & Evaluation │
      └──────────────┘          └──────────────┘
              │                           │
              └─────────────┬─────────────┘
                            ▼
                ┌──────────────────────┐
                │   Performance        │
                │   Metrics & Plots    │
                └──────────────────────┘
```

### Core Components

**1. Feature Extractor**
- Loads audio at 22,050 Hz
- Extracts 13 MFCC coefficients
- Aggregates via mean + standard deviation
- Output: 26-dimensional feature vector

**2. Model Trainer**
- K-Nearest Neighbors (KNN)
- Decision Trees
- Random Forest (ensemble)
- Support Vector Machine (SVM)

**3. Optimizer**
- GridSearchCV with 3-fold cross-validation
- Reduced parameter grids for speed
- Parallel processing (`n_jobs=-1`)

**4. Evaluator**
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices
- Per-genre performance analysis

---

## 📈 Results

### Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | Best Hyperparameters |
|-------|----------|-----------|--------|----------|---------------------|
| **SVM (Tuned)** | **70.0%** 🏆 | **70.3%** | **70.0%** | **70.0%** | C=10, kernel=rbf |
| Random Forest (Tuned) | 65.0% | 64.9% | 65.0% | 64.6% | n_estimators=100, max_depth=20 |
| KNN (Tuned) | 62.5% | 64.2% | 62.5% | 62.7% | n_neighbors=5, weights=distance |
| Decision Tree (Tuned) | 49.5% | 51.4% | 49.5% | 50.1% | max_depth=20, min_samples_split=2 |

### 🎯 Best Model: Support Vector Machine (SVM)
- **Test Accuracy:** 70.0%
- **Cross-Validation Score:** 68.46%
- **Optimal Parameters:** C=10, RBF kernel
- **Training Time:** ~30 seconds

### Key Findings

✅ **Best Model: SVM** - Achieved 70% accuracy with RBF kernel  
✅ **Classical & Metal** - Highest accuracy at 90% each (distinctive timbral signatures)  
✅ **Strong Performance** - Jazz (75%), Pop (75%), Blues (70%)  
⚠️ **Challenging Genres** - Disco (60%), Hiphop (60%), Reggae (55%)  
📊 **Feature Quality** - 26-dim MFCCs provide sufficient spectral resolution

### Per-Genre Performance (SVM Model)

| Genre | Precision | Recall | F1-Score | Notes |
|-------|-----------|--------|----------|-------|
| **Classical** | 0.90 | 0.90 | 0.90 | Best - Distinctive orchestral timbre |
| **Metal** | 0.90 | 0.90 | 0.90 | Best - Heavy distortion signature |
| **Jazz** | 0.71 | 0.75 | 0.73 | Strong - Complex harmonics |
| **Pop** | 0.65 | 0.75 | 0.70 | Good - Modern production |
| **Blues** | 0.74 | 0.70 | 0.72 | Good - Guitar-driven |
| **Country** | 0.72 | 0.65 | 0.68 | Moderate - Overlaps with rock |
| **Rock** | 0.67 | 0.60 | 0.63 | Moderate - Similar to country/pop |
| **Reggae** | 0.65 | 0.55 | 0.59 | Challenging - Rhythmic similarities |
| **Hiphop** | 0.57 | 0.60 | 0.59 | Challenging - Electronic elements |
| **Disco** | 0.52 | 0.60 | 0.56 | Challenging - Overlaps with hiphop |

### Common Misclassifications

- `Rock ↔ Country` - Guitar-driven instrumentation overlap (similar acoustic signatures)
- `Pop ↔ Rock` - Modern production style similarities (electronic processing)
- `Jazz ↔ Blues` - Shared harmonic structures (improvisation patterns)
- `Disco ↔ Hiphop` - Similar rhythmic patterns (electronic dance beats)
- `Country ↔ Rock` - Acoustic guitar and vocal style overlaps

---

## ⚡ Performance

### Expected Runtime

| Phase | Duration |
|-------|----------|
| Feature Extraction | 3-5 minutes |
| Baseline Training | 1-2 minutes |
| Hyperparameter Tuning | 3-5 minutes |
| Evaluation & Visualization | 1-2 minutes |
| **TOTAL** | **~7-10 minutes** |

*Optimized with reduced parameter grids and 3-fold CV*

---

## 🔬 Technical Details

### Feature Engineering

```python
# MFCC Configuration
n_mfcc = 13              # Standard coefficient count
aggregation = mean + std  # Temporal compression
feature_dim = 26         # Final vector size (13×2)
```

### Optimized Hyperparameter Grids

**KNN**
```python
{'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
```

**Decision Tree**
```python
{'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
```

**Random Forest**
```python
{'n_estimators': [50, 100], 'max_depth': [10, 20]}
```

**SVM**
```python
{'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
```

---

## 📁 Repository Structure

```
music-genre-classification/
│
├── Music Genre Final Project.ipynb  # Main implementation
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── .gitignore                       # Git ignore rules
└── GTZAN/                          # Dataset (not in repo)
    └── genres_original/
        ├── blues/
        ├── classical/
        ├── country/
        └── ... (10 genre folders)
```

---

## 🚧 Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Inconsistent audio length | Fixed duration loading with librosa |
| Computational cost | Efficient batch processing + progress tracking |
| Class balance | Stratified train/test splitting |
| Hyperparameter complexity | GridSearchCV with reduced grids |
| Feature dimensionality | Standard MFCC count (13) with mean+std |

---

## ️ Technologies

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat-square)

**Core Libraries:**
- `librosa` - Audio processing and MFCC extraction
- `scikit-learn` - Machine learning models and evaluation
- `pandas` & `numpy` - Data manipulation
- `matplotlib` & `seaborn` - Visualization

---

## 📚 References

- [GTZAN Dataset - Marsyas](http://marsyas.info/downloads/datasets.html)
- [Librosa Documentation](https://librosa.org/)
- [Scikit-learn User Guide](https://scikit-learn.org/)
- [MFCCs Explained](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)

---

Youtube: https://youtu.be/XSfJ4nwvX6o?si=eKnCJI6DdYzz3QBx 

<div align="center">

*Music Genre Classification - AI-Powered Audio Analysis* ⚡

</div>

