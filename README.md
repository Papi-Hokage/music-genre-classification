# Music Genre Classification Using Machine Learning

**CAP 4630 - Introduction to Artificial Intelligence**  
**Fall 2025 - Final Project**  
**Student:** Andres Hernandez  
**Instructor:** Dr. Ahmed Imteaj

---

## 📋 Project Overview

This project implements a supervised machine learning system for classifying music tracks into genres using audio features. The system extracts Mel-Frequency Cepstral Coefficients (MFCCs) from audio files and trains multiple classification models to predict genre labels.

### 🎯 Objectives

- Classify audio tracks into 10 musical genres using MFCC-based features
- Compare performance of multiple machine learning algorithms
- Optimize models through systematic hyperparameter tuning
- Achieve measurable performance with comprehensive evaluation metrics

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

## 🛠️ Technologies Used

- **Python 3.8+**
- **Audio Processing:** librosa
- **Machine Learning:** scikit-learn
- **Data Analysis:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Development:** Jupyter Notebook

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/music-genre-classification.git
cd music-genre-classification
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

Download the GTZAN dataset and place it in the project directory. Your structure should look like:

```
music-genre-classification/
├── Music Genre Final Project.ipynb
├── GTZAN/
│   └── genres_original/
│       ├── blues/
│       ├── classical/
│       ├── country/
│       └── ... (10 genre folders)
├── requirements.txt
├── README.md
└── .gitignore
```

### 5. Update Dataset Path

Open the notebook and update the `DATASET_PATH` variable with your local path.

---

## 📊 Methodology

### Pipeline Overview

1. **Audio Loading** - Load WAV files at 22,050 Hz sampling rate
2. **MFCC Extraction** - Extract 13 MFCC coefficients per audio file
3. **Feature Aggregation** - Compute mean and standard deviation across time frames
4. **Data Splitting** - 80% training, 20% testing with stratification
5. **Model Training** - Train KNN, Decision Tree, Random Forest, SVM
6. **Hyperparameter Tuning** - GridSearchCV with 5-fold cross-validation
7. **Evaluation** - Accuracy, Precision, Recall, F1-Score, Confusion Matrix
8. **Analysis** - Identify misclassifications and performance patterns

### Models Implemented

- **K-Nearest Neighbors (KNN)** - Instance-based learning
- **Decision Tree** - Interpretable decision boundaries
- **Random Forest** - Ensemble method with bootstrap aggregation
- **Support Vector Machine (SVM)** - Maximum margin classifier

---

## 📈 Results

### Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| KNN (Tuned) | TBD | TBD | TBD | TBD |
| Decision Tree (Tuned) | TBD | TBD | TBD | TBD |
| Random Forest (Tuned) | TBD | TBD | TBD | TBD |
| SVM (Tuned) | TBD | TBD | TBD | TBD |

*Note: Results will be populated after running the complete notebook.*

### Key Findings

- MFCC features effectively capture timbral characteristics for genre classification
- Classical and metal genres typically achieve highest accuracy due to distinctive timbral profiles
- Rock, country, and pop often confused due to similar instrumentation
- Hyperparameter tuning provides measurable performance improvements

---

## 🎥 Project Presentation

**[Link to Presentation Video]** *(Add YouTube/Google Drive link here after recording)*

**Duration:** 10 minutes  
**Format:** Technical walkthrough with live demo

---

## 📁 Repository Structure

```
.
├── Music Genre Final Project.ipynb  # Main Jupyter notebook
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── .gitignore                       # Git ignore rules
└── GTZAN/                          # Dataset folder (not in repo)
```

---

## 🏃 Running the Project

### Quick Start

1. Open the notebook:
   ```bash
   jupyter notebook "Music Genre Final Project.ipynb"
   ```

2. Run cells sequentially:
   - **Cell 1-2:** Import libraries and configure dataset path
   - **Cell 3-4:** Load audio files (~2-3 minutes)
   - **Cell 5:** Extract MFCC features (~10-20 minutes)
   - **Cell 6-8:** Exploratory data analysis
   - **Cell 9-11:** Train and evaluate baseline models
   - **Cell 12-14:** Hyperparameter tuning (~15-30 minutes)
   - **Cell 15-17:** Generate visualizations and results

### Expected Runtime

- **Feature Extraction:** 10-20 minutes (CPU-dependent)
- **Baseline Training:** 2-5 minutes
- **Hyperparameter Tuning:** 15-30 minutes
- **Total:** ~30-60 minutes

---

## 🔬 Technical Details

### Feature Engineering

- **MFCCs:** 13 coefficients extracted using librosa
- **Aggregation:** Mean and standard deviation across time frames
- **Final Feature Vector:** 26 dimensions (13 means + 13 stds)

### Hyperparameter Grids

**KNN:**
- n_neighbors: [3, 5, 7, 9, 11]
- weights: ['uniform', 'distance']
- metric: ['euclidean', 'manhattan']

**Decision Tree:**
- max_depth: [5, 10, 15, 20, None]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]

**Random Forest:**
- n_estimators: [50, 100, 200]
- max_depth: [10, 20, None]
- min_samples_split: [2, 5]

**SVM:**
- C: [0.1, 1, 10]
- kernel: ['rbf', 'linear']
- gamma: ['scale', 'auto']

---

## 🚧 Challenges & Solutions

1. **Inconsistent Audio Length** → Fixed duration loading with librosa
2. **Computational Cost** → Efficient batch processing with progress tracking
3. **Class Balance** → Stratified train/test splitting
4. **Hyperparameter Tuning** → GridSearchCV with cross-validation
5. **Feature Dimensionality** → Standard MFCC count (13) with mean+std aggregation

---

## 🔮 Future Work

- **Data Augmentation:** Pitch shifting, time stretching, noise injection
- **Larger Datasets:** FMA, Million Song Dataset, Spotify API
- **Advanced Features:** Mel spectrograms, chroma features, rhythm features
- **Deep Learning:** CNNs on spectrograms, RNNs for temporal modeling
- **Ensemble Methods:** Stacking multiple model predictions
- **Real-Time Classification:** Streaming audio processing
- **Multi-Label Classification:** Hybrid genres and sub-genres

---

## 📚 References

- GTZAN Dataset: [Marsyas](http://marsyas.info/downloads/datasets.html)
- Librosa Documentation: [librosa.org](https://librosa.org/)
- Scikit-learn User Guide: [scikit-learn.org](https://scikit-learn.org/)

---

## 📄 License

This project is for educational purposes as part of CAP 4630 coursework.

---

## 👤 Author

**Andres Hernandez**  
Florida Atlantic University  
CAP 4630 - Introduction to Artificial Intelligence  
Fall 2025

---

## 🙏 Acknowledgments

- Dr. Ahmed Imteaj for course instruction and guidance
- GTZAN dataset creators for providing the music corpus
- Open-source community for librosa and scikit-learn libraries

---

**⭐ If you found this project helpful, please give it a star!**
