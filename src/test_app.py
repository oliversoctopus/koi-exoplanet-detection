"""
Test script to verify the Streamlit app components work correctly
"""

import sys
import os

def test_imports():
    """Test if all required libraries can be imported"""
    print("Testing imports...")
    try:
        import streamlit as st
        print("[OK] Streamlit imported")
    except ImportError as e:
        print(f"[FAIL] Failed to import Streamlit: {e}")
        return False

    try:
        import plotly.express as px
        import plotly.graph_objects as go
        print("[OK] Plotly imported")
    except ImportError as e:
        print(f"[FAIL] Failed to import Plotly: {e}")
        return False

    try:
        import pandas as pd
        import numpy as np
        print("[OK] Pandas and NumPy imported")
    except ImportError as e:
        print(f"[FAIL] Failed to import data libraries: {e}")
        return False

    try:
        import lightgbm as lgb
        print("[OK] LightGBM imported")
    except ImportError as e:
        print(f"[FAIL] Failed to import LightGBM: {e}")
        return False

    return True

def test_model_loading():
    """Test if model files can be loaded"""
    print("\nTesting model loading...")
    import pickle
    import json

    # Check if files exist
    files_to_check = [
        'exoplanet_detector.pkl',
        'exoplanet_detector_metrics.json',
        '../data/preprocessing/preprocessor.pkl'
    ]

    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"[OK] Found: {file_path}")
        else:
            print(f"[FAIL] Missing: {file_path}")
            return False

    # Try loading the model
    try:
        with open('exoplanet_detector.pkl', 'rb') as f:
            model_package = pickle.load(f)
        print("[OK] Model loaded successfully")

        # Check model components
        if 'model' in model_package:
            print("[OK] Model object found")
        else:
            print("[FAIL] Model object not found in package")
            return False

    except Exception as e:
        print(f"[FAIL] Failed to load model: {e}")
        return False

    # Try loading metrics
    try:
        with open('exoplanet_detector_metrics.json', 'r') as f:
            metrics = json.load(f)
        print("[OK] Metrics loaded successfully")
        print(f"  - F1 Score: {metrics['evaluation_metrics']['f1_score']:.4f}")
        print(f"  - ROC AUC: {metrics['evaluation_metrics']['roc_auc']:.4f}")
    except Exception as e:
        print(f"[FAIL] Failed to load metrics: {e}")
        return False

    # Try loading preprocessor
    try:
        with open('../data/preprocessing/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        print("[OK] Preprocessor loaded successfully")
        print(f"  - Features: {len(preprocessor['feature_columns'])}")
    except Exception as e:
        print(f"[FAIL] Failed to load preprocessor: {e}")
        return False

    return True

def test_sample_prediction():
    """Test making a prediction with the model"""
    print("\nTesting sample prediction...")
    import pickle
    import pandas as pd
    import numpy as np

    try:
        # Load model
        with open('exoplanet_detector.pkl', 'rb') as f:
            model_package = pickle.load(f)
        model = model_package['model']

        # Load preprocessor
        with open('../data/preprocessing/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)

        # Create sample data
        sample_features = [10.0, 500.0, 2.0, 800.0, 0.5, 3.0, 30.0, 5800.0, 4.4, 1.0, 14.0, 100.0]
        sample_scaled = preprocessor['scaler'].transform([sample_features])

        # Make prediction
        prediction = model.predict(sample_scaled)[0]
        probability = model.predict_proba(sample_scaled)[0]

        print("[OK] Prediction successful")
        print(f"  - Classification: {'Planet' if prediction == 1 else 'False Positive'}")
        print(f"  - Confidence: {max(probability):.2%}")

        return True

    except Exception as e:
        print(f"[FAIL] Failed to make prediction: {e}")
        return False

def main():
    print("="*60)
    print("NASA EXOPLANET DETECTION - APP COMPONENT TEST")
    print("="*60)

    all_tests_passed = True

    # Run tests
    if not test_imports():
        all_tests_passed = False

    if not test_model_loading():
        all_tests_passed = False

    if not test_sample_prediction():
        all_tests_passed = False

    # Summary
    print("\n" + "="*60)
    if all_tests_passed:
        print("[OK] ALL TESTS PASSED - App is ready to run!")
        print("\nTo start the app, run:")
        print("  streamlit run app.py")
    else:
        print("[FAIL] Some tests failed - Please check the errors above")

    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)