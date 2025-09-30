# 🪐 NASA Exoplanet Detection with AI

**NASA Space Apps Challenge 2025 - A World Away: Hunting for Exoplanets with AI**

An AI-powered system for detecting exoplanets from Kepler Space Telescope transit data using LightGBM machine learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40.2-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Project Overview

This project automates the detection of exoplanets from transit signals in the Kepler Space Telescope data, reducing manual vetting by astronomers. Using supervised machine learning with LightGBM, we classify Kepler Objects of Interest (KOI) as either confirmed/candidate planets or false positives.

### Key Features
- **High-Performance ML Model**: LightGBM classifier achieving 83% F1-score and 92.6% ROC-AUC, along with a TabNet and Random Forest classifier with similar levels of accuracy.
- **Interactive Web Application**: Streamlit-based UI for real-time predictions
- **Comprehensive Data Pipeline**: Automated preprocessing, feature engineering, and model training
- **Visualization Dashboard**: Interactive plots for data exploration and model interpretation

## 📊 Model Performance (LightGBM)

| Metric | Score |
|--------|-------|
| **F1-Score** | 83.16% |
| **ROC-AUC** | 92.56% |
| **Precision** | 82.72% |
| **Recall** | 83.60% |
| **Accuracy** | 83.27% |

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/nasa-exoplanet-detection.git
cd nasa-exoplanet-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
cd src
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
nasa-exoplanet-detection/
│
├── data/                          # Data directory
│   ├── kepler_koi.csv           # Raw Kepler KOI dataset
│   └── preprocessing/            # Processed data and visualizations
│       ├── X_train.csv          # Training features
│       ├── X_val.csv            # Validation features
│       ├── X_test.csv           # Test features
│       ├── y_train.csv          # Training labels
│       ├── y_val.csv            # Validation labels
│       ├── y_test.csv           # Test labels
│       ├── preprocessor.pkl     # Saved preprocessor
│       └── *.png                # Generated visualizations
│
├── src/                          # Source code
│   ├── model.py                 # Model training script
│   ├── app.py                   # Streamlit web application
│   ├── exoplanet_detector.pkl   # Trained model
│   └── *.json                   # Model metrics
│
├── CLAUDE.md                     # Project specifications
├── requirements.txt              # Python dependencies
└── README.md                    # This file
```

## 🔬 Dataset

- **Source**: NASA Exoplanet Archive - Kepler Objects of Interest (KOI)
- **Size**: 9,564 transit signals
- **Features**: 12 key astronomical features
- **Target**: Binary classification (Planet vs False Positive)
- **Class Distribution**: ~49% planets, ~51% false positives

### Key Features Used

1. **koi_period**: Orbital period (days)
2. **koi_depth**: Transit depth (ppm)
3. **koi_prad**: Planet radius (Earth radii)
4. **koi_teq**: Equilibrium temperature (K)
5. **koi_impact**: Impact parameter
6. **koi_duration**: Transit duration (hours)
7. **koi_model_snr**: Model signal-to-noise ratio
8. **koi_steff**: Stellar effective temperature (K)
9. **koi_slogg**: Stellar surface gravity
10. **koi_srad**: Stellar radius (Solar radii)
11. **koi_kepmag**: Kepler magnitude
12. **koi_insol**: Insolation flux

## 🎯 Usage

### Web Application

1. **Launch the app**: Run `streamlit run app.py` from the `src` directory
2. **Choose input method**:
   - **Manual Input**: Enter transit parameters manually
   - **Sample Data**: Test with pre-loaded samples
   - **Upload CSV**: Upload your own transit data

3. **View results**:
   - Classification (Planet/False Positive)
   - Confidence score
   - Probability distribution
   - Feature importance

### Python API

```python
import pickle
import pandas as pd

# Load model
with open('exoplanet_detector.pkl', 'rb') as f:
    model_package = pickle.load(f)

model = model_package['model']

# Prepare your data (12 features)
features = pd.DataFrame({
    'koi_period': [10.0],
    'koi_depth': [500.0],
    # ... add all 12 features
})

# Make prediction
prediction = model.predict(features)
probability = model.predict_proba(features)

print(f"Classification: {'Planet' if prediction[0] == 1 else 'False Positive'}")
print(f"Confidence: {max(probability[0]):.2%}")
```

## 🔧 Training the Model

To retrain the model with new data or parameters:

1. **Prepare data**: Place your KOI dataset in `data/kepler_koi.csv`
2. **Preprocess**: Run `python data/preprocessing/data_preprocessing.py`
3. **Train model**: Run `python src/model.py`

The trained model will be saved as `exoplanet_detector.pkl`

## 📈 Model Architecture

- **Algorithm**: LightGBM (Light Gradient Boosting Machine)
- **Type**: Binary classification
- **Key Hyperparameters**:
  - n_estimators: 200
  - learning_rate: 0.1
  - max_depth: 8
  - num_leaves: 31
  - subsample: 0.8

## 🏆 NASA Space Apps Challenge 2025

This project was developed as a proof of concept before the NASA Space Apps Challenge 2025, addressing the challenge:
**"A World Away: Hunting for Exoplanets with AI"**

### Challenge Goals Met
✅ Automated exoplanet detection from transit data
✅ High-accuracy machine learning model (83% F1-score)
✅ Interactive visualization and exploration tools
✅ Open-source implementation with documentation
✅ User-friendly web interface for predictions

## 📚 References

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [Kepler Mission](https://www.nasa.gov/mission_pages/kepler/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification

### Citation

If you use this work, please cite:
```
NASA Exoplanet Archive (2025)
DOI: http://doi.org/10.17616/R3X31K
```

## 📄 License

This project is open source and available under the MIT License.

## 📧 Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

**Note**: This project achieves 83% F1-score, which is excellent for real-world Kepler data considering the inherent noise and ambiguity in transit signals. The high ROC-AUC of 92.6% demonstrates strong discriminative ability between planets and false positives.