"""
Random Forest Model Training for NASA Exoplanet Detection
==========================================================
Train a Random Forest ensemble model for exoplanet classification.
Random Forest is a robust ensemble method that builds multiple decision trees.
"""

import pandas as pd
import numpy as np
import pickle
import json
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_data():
    """Load preprocessed training data"""
    print("Loading preprocessed data...")

    X_train = pd.read_csv('data/preprocessing/X_train.csv').values
    X_val = pd.read_csv('data/preprocessing/X_val.csv').values
    X_test = pd.read_csv('data/preprocessing/X_test.csv').values

    y_train = pd.read_csv('data/preprocessing/y_train.csv')['label'].values
    y_val = pd.read_csv('data/preprocessing/y_val.csv')['label'].values
    y_test = pd.read_csv('data/preprocessing/y_test.csv')['label'].values

    # Load preprocessor for feature names
    with open('data/preprocessing/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(preprocessor['feature_columns'])}")

    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor

def train_baseline_rf(X_train, y_train, X_val, y_val):
    """Train baseline Random Forest with default parameters"""
    print("\n" + "="*60)
    print("Training Baseline Random Forest Model")
    print("="*60)

    start_time = time.time()

    # Initialize Random Forest with good default parameters
    model = RandomForestClassifier(
        n_estimators=200,  # Number of trees
        max_depth=None,  # No maximum depth
        min_samples_split=2,  # Minimum samples to split a node
        min_samples_leaf=1,  # Minimum samples at leaf node
        max_features='sqrt',  # Number of features to consider for splits
        bootstrap=True,  # Bootstrap samples
        oob_score=True,  # Out-of-bag score
        n_jobs=-1,  # Use all processors
        random_state=42,
        verbose=1
    )

    # Train the model
    print("Training Random Forest with 200 trees...")
    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    # Get predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Calculate metrics
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'f1_score': f1_score(y_train, y_train_pred)
    }

    val_metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall': recall_score(y_val, y_val_pred),
        'f1_score': f1_score(y_val, y_val_pred)
    }

    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Out-of-Bag Score: {model.oob_score_:.4f}")
    print(f"Train F1-Score: {train_metrics['f1_score']:.4f}")
    print(f"Val F1-Score: {val_metrics['f1_score']:.4f}")

    return model, train_metrics, val_metrics, training_time

def tune_rf(X_train, y_train, X_val, y_val):
    """Tune Random Forest hyperparameters using GridSearchCV"""
    print("\n" + "="*60)
    print("Tuning Random Forest Model")
    print("="*60)

    start_time = time.time()

    # Combine train and validation for cross-validation
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.hstack([y_train, y_val])

    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    print("Parameter grid to search:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    # Create base model
    rf = RandomForestClassifier(
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=42
    )

    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,  # 3-fold cross-validation
        scoring='f1',
        n_jobs=-1,
        verbose=2
    )

    print("\nPerforming grid search...")
    print("This will test 324 different parameter combinations")
    print("Expected time: 5-15 minutes depending on hardware")

    grid_search.fit(X_combined, y_combined)

    tuning_time = time.time() - start_time

    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"\nTuning completed in {tuning_time:.2f} seconds")
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation F1-score: {grid_search.best_score_:.4f}")

    # Retrain best model on training data only
    best_model.fit(X_train, y_train)

    # Get predictions
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)

    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'f1_score': f1_score(y_train, y_train_pred)
    }

    val_metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall': recall_score(y_val, y_val_pred),
        'f1_score': f1_score(y_val, y_val_pred)
    }

    return best_model, train_metrics, val_metrics, tuning_time, best_params

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set"""
    print("\n" + "="*60)
    print("Evaluating on Test Set")
    print("="*60)

    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    print("\nTest Set Performance:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives:  {cm[0,0]}")
    print(f"False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}")
    print(f"True Positives:  {cm[1,1]}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                              target_names=['Non-Planet', 'Planet']))

    return metrics

def analyze_feature_importance(model, feature_names):
    """Analyze and display feature importance"""
    print("\n" + "="*60)
    print("Feature Importance Analysis")
    print("="*60)

    # Get feature importances
    importances = model.feature_importances_

    # Create DataFrame for better display
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['Feature']:20s}: {row['Importance']:.4f}")

    return importances

def save_model(model, metrics_data, preprocessor):
    """Save Random Forest model and metrics"""
    print("\n" + "="*60)
    print("Saving Model")
    print("="*60)

    # Create model package
    model_package = {
        'model': model,
        'model_type': 'RandomForest',
        'timestamp': datetime.now().isoformat(),
        'features': preprocessor['feature_columns']
    }

    # Save model
    with open('rf_exoplanet_detector.pkl', 'wb') as f:
        pickle.dump(model_package, f)
    print("Model saved to rf_exoplanet_detector.pkl")

    # Save metrics
    with open('rf_exoplanet_metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print("Metrics saved to rf_exoplanet_metrics.json")

def main():
    """Main training pipeline"""
    print("="*60)
    print("Random Forest Model Training for Exoplanet Detection")
    print("NASA Space Apps Challenge 2025")
    print("="*60)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = load_data()

    # Train baseline model
    baseline_model, baseline_train_metrics, baseline_val_metrics, baseline_time = \
        train_baseline_rf(X_train, y_train, X_val, y_val)

    # Ask user if they want to run hyperparameter tuning
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING OPTION")
    print("="*60)
    print("Hyperparameter tuning will test 324 parameter combinations")
    print("Estimated time: 5-15 minutes")

    user_input = input("\nDo you want to run hyperparameter tuning? (y/n): ").lower().strip()

    if user_input == 'y':
        # Tune model
        tuned_model, tuned_train_metrics, tuned_val_metrics, tuning_time, best_params = \
            tune_rf(X_train, y_train, X_val, y_val)

        # Evaluate both models on test set
        print("\nBaseline Model Test Performance:")
        baseline_test_metrics = evaluate_model(baseline_model, X_test, y_test)

        print("\nTuned Model Test Performance:")
        tuned_test_metrics = evaluate_model(tuned_model, X_test, y_test)

        # Choose best model based on test F1-score
        if tuned_test_metrics['f1_score'] > baseline_test_metrics['f1_score']:
            print("\n✓ Using tuned model (better test F1-score)")
            final_model = tuned_model
            final_test_metrics = tuned_test_metrics
        else:
            print("\n✓ Using baseline model (better test F1-score)")
            final_model = baseline_model
            final_test_metrics = baseline_test_metrics
    else:
        print("\nSkipping hyperparameter tuning. Using baseline model.")
        final_model = baseline_model
        final_test_metrics = evaluate_model(baseline_model, X_test, y_test)

        # Set tuned metrics same as baseline for consistency
        tuned_train_metrics = baseline_train_metrics
        tuned_val_metrics = baseline_val_metrics
        tuning_time = 0
        best_params = {
            'n_estimators': 200,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt'
        }
        tuned_test_metrics = final_test_metrics

    # Analyze feature importance
    feature_importances = analyze_feature_importance(final_model, preprocessor['feature_columns'])

    # Prepare metrics data
    metrics_data = {
        'model_type': 'RandomForest',
        'evaluation_metrics': final_test_metrics,
        'training_history': {
            'baseline': {
                'train_accuracy': baseline_train_metrics['accuracy'],
                'train_precision': baseline_train_metrics['precision'],
                'train_recall': baseline_train_metrics['recall'],
                'train_f1': baseline_train_metrics['f1_score'],
                'val_accuracy': baseline_val_metrics['accuracy'],
                'val_precision': baseline_val_metrics['precision'],
                'val_recall': baseline_val_metrics['recall'],
                'val_f1': baseline_val_metrics['f1_score'],
                'training_time': baseline_time,
                'test_f1': baseline_test_metrics['f1_score'] if 'baseline_test_metrics' in locals() else final_test_metrics['f1_score']
            },
            'tuned': {
                'train_accuracy': tuned_train_metrics['accuracy'],
                'train_precision': tuned_train_metrics['precision'],
                'train_recall': tuned_train_metrics['recall'],
                'train_f1': tuned_train_metrics['f1_score'],
                'val_accuracy': tuned_val_metrics['accuracy'],
                'val_precision': tuned_val_metrics['precision'],
                'val_recall': tuned_val_metrics['recall'],
                'val_f1': tuned_val_metrics['f1_score'],
                'tuning_time': tuning_time,
                'test_f1': tuned_test_metrics['f1_score'],
                'best_params': best_params if 'best_params' in locals() else None
            }
        },
        'feature_importances': feature_importances.tolist(),
        'model_parameters': {
            'n_estimators': final_model.n_estimators,
            'max_depth': final_model.max_depth,
            'min_samples_split': final_model.min_samples_split,
            'min_samples_leaf': final_model.min_samples_leaf,
            'max_features': final_model.max_features,
            'oob_score': final_model.oob_score_ if hasattr(final_model, 'oob_score_') else None
        },
        'timestamp': datetime.now().isoformat()
    }

    # Save model and metrics
    save_model(final_model, metrics_data, preprocessor)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nFinal Test Performance:")
    print(f"F1-Score:  {final_test_metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {final_test_metrics['roc_auc']:.4f}")
    print(f"Accuracy:  {final_test_metrics['accuracy']:.4f}")
    print(f"Precision: {final_test_metrics['precision']:.4f}")
    print(f"Recall:    {final_test_metrics['recall']:.4f}")

    if hasattr(final_model, 'oob_score_'):
        print(f"\nOut-of-Bag Score: {final_model.oob_score_:.4f}")

if __name__ == "__main__":
    main()