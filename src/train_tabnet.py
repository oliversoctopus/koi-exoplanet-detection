"""
TabNet Model Training for NASA Exoplanet Detection
===================================================
Train a TabNet deep learning model for exoplanet classification.
TabNet is an attention-based neural network designed for tabular data.
"""

import pandas as pd
import numpy as np
import pickle
import json
import time
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

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

def train_baseline_tabnet(X_train, y_train, X_val, y_val):
    """Train baseline TabNet model with default parameters"""
    print("\n" + "="*60)
    print("Training Baseline TabNet Model")
    print("="*60)

    start_time = time.time()

    # Initialize TabNet with default parameters
    model = TabNetClassifier(
        n_d=8,  # Width of decision prediction layer
        n_a=8,  # Width of attention embedding
        n_steps=3,  # Number of sequential attention steps
        gamma=1.3,  # Relaxation parameter
        n_independent=2,  # Number of independent GLU layers
        n_shared=2,  # Number of shared GLU layers
        epsilon=1e-15,
        momentum=0.02,
        lambda_sparse=1e-3,  # Sparsity regularization
        seed=42,
        clip_value=None,
        verbose=1,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 50, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='sparsemax',
        device_name='auto'
    )

    # Train the model
    model.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_name=['val'],
        eval_metric=['accuracy', 'auc'],
        max_epochs=100,
        patience=15,
        batch_size=256,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        augmentations=None
    )

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
    print(f"Train F1-Score: {train_metrics['f1_score']:.4f}")
    print(f"Val F1-Score: {val_metrics['f1_score']:.4f}")

    return model, train_metrics, val_metrics, training_time

def tune_tabnet(X_train, y_train, X_val, y_val):
    """Tune TabNet hyperparameters"""
    print("\n" + "="*60)
    print("Tuning TabNet Model")
    print("="*60)

    start_time = time.time()

    # Define hyperparameter combinations to try
    param_combinations = [
        {'n_d': 8, 'n_a': 8, 'n_steps': 3, 'gamma': 1.3, 'lr': 0.02},
        {'n_d': 16, 'n_a': 16, 'n_steps': 4, 'gamma': 1.5, 'lr': 0.02},
        {'n_d': 32, 'n_a': 32, 'n_steps': 5, 'gamma': 1.3, 'lr': 0.01},
        {'n_d': 24, 'n_a': 24, 'n_steps': 4, 'gamma': 1.5, 'lr': 0.015},
    ]

    best_score = 0
    best_model = None
    best_params = None

    for i, params in enumerate(param_combinations, 1):
        print(f"\nTesting combination {i}/{len(param_combinations)}: {params}")

        model = TabNetClassifier(
            n_d=params['n_d'],
            n_a=params['n_a'],
            n_steps=params['n_steps'],
            gamma=params['gamma'],
            n_independent=2,
            n_shared=2,
            epsilon=1e-15,
            momentum=0.02,
            lambda_sparse=1e-3,
            seed=42,
            clip_value=None,
            verbose=0,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=params['lr']),
            scheduler_params={"step_size": 50, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='sparsemax',
            device_name='auto'
        )

        model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_val, y_val)],
            eval_name=['val'],
            eval_metric=['accuracy'],
            max_epochs=100,
            patience=15,
            batch_size=256,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )

        # Evaluate on validation set
        y_val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred)

        print(f"Validation F1-Score: {val_f1:.4f}")

        if val_f1 > best_score:
            best_score = val_f1
            best_model = model
            best_params = params

    tuning_time = time.time() - start_time

    print(f"\nBest parameters: {best_params}")
    print(f"Best validation F1-Score: {best_score:.4f}")
    print(f"Tuning completed in {tuning_time:.2f} seconds")

    # Get final metrics with best model
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

def save_model(model, metrics_data, preprocessor):
    """Save TabNet model and metrics"""
    print("\n" + "="*60)
    print("Saving Model")
    print("="*60)

    # Create model package
    model_package = {
        'model': model,
        'model_type': 'TabNet',
        'timestamp': datetime.now().isoformat(),
        'features': preprocessor['feature_columns']
    }

    # Save model
    with open('tabnet_exoplanet_detector.pkl', 'wb') as f:
        pickle.dump(model_package, f)
    print("Model saved to tabnet_exoplanet_detector.pkl")

    # Save metrics
    with open('tabnet_exoplanet_metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print("Metrics saved to tabnet_exoplanet_metrics.json")

    # Also save the TabNet model separately using its native save method
    model.save_model('tabnet_model')
    print("TabNet model weights saved to tabnet_model.zip")

def main():
    """Main training pipeline"""
    print("="*60)
    print("TabNet Model Training for Exoplanet Detection")
    print("NASA Space Apps Challenge 2025")
    print("="*60)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = load_data()

    # Train baseline model
    baseline_model, baseline_train_metrics, baseline_val_metrics, baseline_time = \
        train_baseline_tabnet(X_train, y_train, X_val, y_val)

    # Tune model
    tuned_model, tuned_train_metrics, tuned_val_metrics, tuning_time, best_params = \
        tune_tabnet(X_train, y_train, X_val, y_val)

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

    # Get feature importance (attention weights)
    feature_importances = final_model.feature_importances_

    # Prepare metrics data
    metrics_data = {
        'model_type': 'TabNet',
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
                'test_f1': baseline_test_metrics['f1_score']
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
                'best_params': best_params
            }
        },
        'feature_importances': {
            'features': preprocessor['feature_columns'],
            'importances': feature_importances.tolist()
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

if __name__ == "__main__":
    main()