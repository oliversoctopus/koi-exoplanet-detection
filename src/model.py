"""
LightGBM Model Training for NASA Kepler Exoplanet Detection
===========================================================
Trains a LightGBM classifier to detect exoplanets from Kepler transit data.
Includes baseline training, hyperparameter tuning, and comprehensive evaluation.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report, roc_curve)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import warnings
import time
warnings.filterwarnings('ignore')

class ExoplanetDetector:
    """LightGBM-based exoplanet detection model"""

    def __init__(self, random_state=42):
        """Initialize the detector"""
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.training_history = {}
        self.evaluation_metrics = {}

    def load_data(self):
        """Load preprocessed data"""
        print("Loading preprocessed data...")

        # Load features
        X_train = pd.read_csv('../data/preprocessing/X_train.csv')
        X_val = pd.read_csv('../data/preprocessing/X_val.csv')
        X_test = pd.read_csv('../data/preprocessing/X_test.csv')

        # Load labels
        y_train = pd.read_csv('../data/preprocessing/y_train.csv')['label']
        y_val = pd.read_csv('../data/preprocessing/y_val.csv')['label']
        y_test = pd.read_csv('../data/preprocessing/y_test.csv')['label']

        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Val: {X_val.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_baseline(self, X_train, y_train, X_val, y_val):
        """Train baseline LightGBM model with default parameters"""
        print("\n" + "="*60)
        print("TRAINING BASELINE MODEL")
        print("="*60)

        # Calculate class weights for imbalanced data
        n_pos = sum(y_train == 1)
        n_neg = sum(y_train == 0)
        scale_pos_weight = n_neg / n_pos

        print(f"Class balance - Positive: {n_pos}, Negative: {n_neg}")
        print(f"Scale positive weight: {scale_pos_weight:.3f}")

        # Create baseline model with sensible defaults
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            verbosity=-1
        )

        # Train with early stopping
        print("\nTraining with early stopping...")
        start_time = time.time()

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=20),
                      lgb.log_evaluation(period=0)]
        )

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Evaluate baseline
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        train_f1 = f1_score(y_train, train_pred)
        val_f1 = f1_score(y_val, val_pred)

        print(f"\nBaseline Performance:")
        print(f"  Train F1-score: {train_f1:.4f}")
        print(f"  Val F1-score: {val_f1:.4f}")

        self.training_history['baseline'] = {
            'train_f1': train_f1,
            'val_f1': val_f1,
            'training_time': training_time
        }

        return self.model

    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val, search_type='grid'):
        """Perform hyperparameter tuning"""
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING")
        print("="*60)

        # Calculate scale_pos_weight
        n_pos = sum(y_train == 1)
        n_neg = sum(y_train == 0)
        scale_pos_weight = n_neg / n_pos

        # Define parameter grid
        if search_type == 'grid':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [4, 6, 8],
                'num_leaves': [20, 31, 40],
                'min_child_samples': [10, 20, 30],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
        else:  # Random search with wider range
            param_grid = {
                'n_estimators': [50, 100, 200, 300, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'max_depth': [3, 4, 5, 6, 7, 8, 10],
                'num_leaves': [15, 20, 31, 40, 50],
                'min_child_samples': [5, 10, 20, 30, 40],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.01, 0.1, 0.5],
                'reg_lambda': [0, 0.01, 0.1, 0.5]
            }

        # Base model
        lgb_base = lgb.LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            verbosity=-1
        )

        # Perform search
        print(f"Performing {search_type} search...")
        print(f"Parameter combinations to test: ", end="")

        if search_type == 'grid':
            # Calculate total combinations for grid search
            n_combinations = 1
            for param_values in param_grid.values():
                n_combinations *= len(param_values)
            print(f"{n_combinations}")

            # Use fewer combinations for faster training
            param_grid_reduced = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [6, 8],
                'num_leaves': [31, 40],
                'min_child_samples': [20],
                'subsample': [0.8],
                'colsample_bytree': [0.8]
            }

            search = GridSearchCV(
                lgb_base, param_grid_reduced, cv=3,
                scoring='f1', n_jobs=-1, verbose=1
            )
        else:
            print("100 (random)")
            search = RandomizedSearchCV(
                lgb_base, param_grid, n_iter=100, cv=3,
                scoring='f1', n_jobs=-1, verbose=1,
                random_state=self.random_state
            )

        start_time = time.time()
        search.fit(X_train, y_train)
        tuning_time = time.time() - start_time

        print(f"\nTuning completed in {tuning_time:.2f} seconds")
        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV F1-score: {search.best_score_:.4f}")

        # Train final model with best parameters and early stopping
        self.best_params = search.best_params_
        self.model = lgb.LGBMClassifier(
            **self.best_params,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            verbosity=-1
        )

        print("\nTraining final model with best parameters...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=20),
                      lgb.log_evaluation(period=0)]
        )

        # Evaluate tuned model
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        train_f1 = f1_score(y_train, train_pred)
        val_f1 = f1_score(y_val, val_pred)

        print(f"\nTuned Model Performance:")
        print(f"  Train F1-score: {train_f1:.4f}")
        print(f"  Val F1-score: {val_f1:.4f}")

        self.training_history['tuned'] = {
            'train_f1': train_f1,
            'val_f1': val_f1,
            'tuning_time': tuning_time,
            'best_params': self.best_params
        }

        return self.model

    def evaluate(self, X_test, y_test, X_train=None, y_train=None):
        """Comprehensive model evaluation"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)

        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Get predictions and probabilities
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        print("\nTest Set Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives: {cm[0,0]}")
        print(f"  False Positives: {cm[0,1]}")
        print(f"  False Negatives: {cm[1,0]}")
        print(f"  True Positives: {cm[1,1]}")

        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred,
                                   target_names=['Non-Planet', 'Planet']))

        # Store metrics
        self.evaluation_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist()
        }

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 5 Most Important Features:")
        for idx, row in feature_importance.head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")

        self.evaluation_metrics['feature_importance'] = feature_importance.to_dict()

        return self.evaluation_metrics

    def plot_results(self, X_test, y_test):
        """Generate visualization plots"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
                   xticklabels=['Non-Planet', 'Planet'],
                   yticklabels=['Non-Planet', 'Planet'])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_ylabel('True Label')
        axes[0,0].set_xlabel('Predicted Label')

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        axes[0,1].plot(fpr, tpr, color='darkorange', lw=2,
                      label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0,1].set_xlim([0.0, 1.0])
        axes[0,1].set_ylim([0.0, 1.05])
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend(loc="lower right")

        # 3. Feature Importance
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=True).tail(10)

        axes[0,2].barh(range(len(feature_importance)),
                      feature_importance['importance'], color='steelblue')
        axes[0,2].set_yticks(range(len(feature_importance)))
        axes[0,2].set_yticklabels(feature_importance['feature'])
        axes[0,2].set_xlabel('Importance')
        axes[0,2].set_title('Top 10 Feature Importances')

        # 4. Prediction Distribution
        axes[1,0].hist(y_prob[y_test==0], bins=30, alpha=0.5,
                      label='Non-Planet', color='red', density=True)
        axes[1,0].hist(y_prob[y_test==1], bins=30, alpha=0.5,
                      label='Planet', color='green', density=True)
        axes[1,0].set_xlabel('Predicted Probability')
        axes[1,0].set_ylabel('Density')
        axes[1,0].set_title('Prediction Probability Distribution')
        axes[1,0].legend()
        axes[1,0].axvline(x=0.5, color='black', linestyle='--', alpha=0.5)

        # 5. Calibration Plot
        from sklearn.calibration import calibration_curve
        fraction_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
        axes[1,1].plot(mean_pred, fraction_pos, marker='o', linewidth=1, label='Model')
        axes[1,1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
        axes[1,1].set_xlabel('Mean Predicted Probability')
        axes[1,1].set_ylabel('Fraction of Positives')
        axes[1,1].set_title('Calibration Plot')
        axes[1,1].legend()

        # 6. Performance Metrics Bar Chart
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        }

        axes[1,2].bar(range(len(metrics)), list(metrics.values()),
                     color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        axes[1,2].set_xticks(range(len(metrics)))
        axes[1,2].set_xticklabels(list(metrics.keys()), rotation=45, ha='right')
        axes[1,2].set_ylabel('Score')
        axes[1,2].set_ylim([0, 1])
        axes[1,2].set_title('Performance Metrics')
        axes[1,2].axhline(y=0.95, color='red', linestyle='--', alpha=0.5,
                         label='Target: 95%')
        axes[1,2].legend()

        # Add values on bars
        for i, v in enumerate(metrics.values()):
            axes[1,2].text(i, v + 0.01, f'{v:.3f}', ha='center')

        plt.suptitle('Exoplanet Detection Model Performance', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("  Visualizations saved to model_performance.png")

    def save_model(self, filepath='exoplanet_detector.pkl'):
        """Save the trained model and metadata"""
        print("\n" + "="*60)
        print("SAVING MODEL")
        print("="*60)

        model_package = {
            'model': self.model,
            'best_params': self.best_params,
            'training_history': self.training_history,
            'evaluation_metrics': self.evaluation_metrics
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)

        print(f"  Model saved to {filepath}")

        # Save metrics as JSON for easy reading
        metrics_filepath = filepath.replace('.pkl', '_metrics.json')
        with open(metrics_filepath, 'w') as f:
            json.dump({
                'best_params': self.best_params,
                'training_history': self.training_history,
                'evaluation_metrics': {k: v for k, v in self.evaluation_metrics.items()
                                      if k != 'feature_importance'}
            }, f, indent=2)

        print(f"  Metrics saved to {metrics_filepath}")

    def load_model(self, filepath='exoplanet_detector.pkl'):
        """Load a saved model"""
        with open(filepath, 'rb') as f:
            model_package = pickle.load(f)

        self.model = model_package['model']
        self.best_params = model_package.get('best_params')
        self.training_history = model_package.get('training_history', {})
        self.evaluation_metrics = model_package.get('evaluation_metrics', {})

        print(f"Model loaded from {filepath}")

def main():
    """Main training pipeline"""
    print("="*60)
    print("NASA EXOPLANET DETECTION MODEL TRAINING")
    print("="*60)

    # Initialize detector
    detector = ExoplanetDetector(random_state=42)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = detector.load_data()

    # Train baseline model
    detector.train_baseline(X_train, y_train, X_val, y_val)

    # Hyperparameter tuning (using grid search for faster results)
    detector.hyperparameter_tuning(X_train, y_train, X_val, y_val, search_type='grid')

    # Evaluate on test set
    detector.evaluate(X_test, y_test, X_train, y_train)

    # Generate visualizations
    detector.plot_results(X_test, y_test)

    # Save model
    detector.save_model('exoplanet_detector.pkl')

    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nFinal Model Performance:")
    print(f"  F1-score: {detector.evaluation_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC: {detector.evaluation_metrics['roc_auc']:.4f}")

    if detector.evaluation_metrics['f1_score'] >= 0.95:
        print("\n[SUCCESS] Target of >95% F1-score achieved!")
    else:
        print(f"\n[INFO] Current F1-score: {detector.evaluation_metrics['f1_score']:.4f}")
        print("       Consider additional tuning or feature engineering")

    print("\nModel saved to: exoplanet_detector.pkl")
    print("Visualizations saved to: model_performance.png")

if __name__ == "__main__":
    main()