"""
Improved LightGBM Model Training for NASA Kepler Exoplanet Detection
====================================================================
Enhanced version with better hyperparameter tuning and feature engineering
to achieve >95% F1-score target.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import pickle
import json
import warnings
import time
warnings.filterwarnings('ignore')

class ImprovedExoplanetDetector:
    """Enhanced LightGBM-based exoplanet detection model"""

    def __init__(self, random_state=42):
        """Initialize the detector"""
        self.random_state = random_state
        self.model = None
        self.best_params = None
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

    def engineer_features(self, X):
        """Create additional engineered features"""
        print("Engineering additional features...")
        X_eng = X.copy()

        # Ratio features
        X_eng['depth_period_ratio'] = X_eng['koi_depth'] / (X_eng['koi_period'] + 1)
        X_eng['depth_duration_ratio'] = X_eng['koi_depth'] / (X_eng['koi_duration'] + 1)
        X_eng['period_duration_ratio'] = X_eng['koi_period'] / (X_eng['koi_duration'] + 1)

        # Interaction features
        X_eng['depth_impact_product'] = X_eng['koi_depth'] * X_eng['koi_impact']
        X_eng['snr_depth_product'] = X_eng['koi_model_snr'] * X_eng['koi_depth']
        X_eng['teq_prad_product'] = X_eng['koi_teq'] * X_eng['koi_prad']

        # Log transformations for skewed features
        for col in ['koi_period', 'koi_depth', 'koi_prad', 'koi_model_snr']:
            if col in X_eng.columns:
                X_eng[f'{col}_log'] = np.log1p(X_eng[col])

        # Polynomial features for important variables
        X_eng['koi_depth_squared'] = X_eng['koi_depth'] ** 2
        X_eng['koi_period_squared'] = X_eng['koi_period'] ** 2

        print(f"  Created {X_eng.shape[1] - X.shape[1]} new features")
        return X_eng

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val):
        """Extensive hyperparameter optimization"""
        print("\n" + "="*60)
        print("ENHANCED HYPERPARAMETER OPTIMIZATION")
        print("="*60)

        # Calculate class weights
        n_pos = sum(y_train == 1)
        n_neg = sum(y_train == 0)
        scale_pos_weight = n_neg / n_pos

        # Extended parameter space
        param_space = {
            'n_estimators': [300, 500, 700, 1000, 1500],
            'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
            'max_depth': [3, 4, 5, 6, 7, 8, 10, 12],
            'num_leaves': [15, 20, 25, 31, 40, 50, 63, 80],
            'min_child_samples': [5, 10, 15, 20, 30, 40],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.001, 0.01, 0.1, 0.3, 0.5],
            'reg_lambda': [0, 0.001, 0.01, 0.1, 0.3, 0.5],
            'min_split_gain': [0, 0.001, 0.01, 0.1],
            'subsample_freq': [0, 1, 3, 5, 7],
            'boosting_type': ['gbdt', 'dart', 'goss']
        }

        # Base model with fixed parameters
        lgb_base = lgb.LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            objective='binary',
            metric='binary_logloss',
            verbosity=-1,
            n_jobs=-1,
            importance_type='gain'
        )

        # Stratified K-Fold for better CV
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        # Random search with more iterations
        search = RandomizedSearchCV(
            lgb_base,
            param_space,
            n_iter=200,  # More iterations for better exploration
            cv=cv_strategy,
            scoring='f1',
            n_jobs=-1,
            verbose=1,
            random_state=self.random_state
        )

        print("Running randomized search with 200 iterations...")
        print("This may take several minutes...")
        start_time = time.time()

        # Combine train and validation for CV
        X_combined = pd.concat([X_train, X_val])
        y_combined = pd.concat([y_train, y_val])

        search.fit(X_combined, y_combined)
        tuning_time = time.time() - start_time

        print(f"\nOptimization completed in {tuning_time/60:.2f} minutes")
        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV F1-score: {search.best_score_:.4f}")

        # Train final model with best parameters
        self.best_params = search.best_params_
        self.model = lgb.LGBMClassifier(
            **self.best_params,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            objective='binary',
            metric='binary_logloss',
            verbosity=-1,
            n_jobs=-1,
            importance_type='gain'
        )

        # Train with early stopping on full training set
        print("\nTraining final model with optimized parameters...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['auc', 'binary_logloss'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)
            ]
        )

        # Evaluate
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        train_f1 = f1_score(y_train, train_pred)
        val_f1 = f1_score(y_val, val_pred)

        print(f"\nOptimized Model Performance:")
        print(f"  Train F1-score: {train_f1:.4f}")
        print(f"  Val F1-score: {val_f1:.4f}")

        return self.model

    def train_ensemble(self, X_train, y_train, X_val, y_val):
        """Train multiple models with different seeds for ensemble"""
        print("\n" + "="*60)
        print("TRAINING ENSEMBLE MODELS")
        print("="*60)

        n_pos = sum(y_train == 1)
        n_neg = sum(y_train == 0)
        scale_pos_weight = n_neg / n_pos

        # Use best parameters if available
        if self.best_params is None:
            # Default good parameters
            base_params = {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 6,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1
            }
        else:
            base_params = self.best_params.copy()

        ensemble_models = []
        ensemble_scores = []

        # Train 5 models with different seeds
        for i in range(5):
            print(f"\nTraining model {i+1}/5...")
            model = lgb.LGBMClassifier(
                **base_params,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state + i,
                objective='binary',
                verbosity=-1,
                n_jobs=-1
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30),
                    lgb.log_evaluation(period=0)
                ]
            )

            val_pred = model.predict(X_val)
            val_f1 = f1_score(y_val, val_pred)

            ensemble_models.append(model)
            ensemble_scores.append(val_f1)
            print(f"  Model {i+1} Val F1-score: {val_f1:.4f}")

        # Store the best single model
        best_idx = np.argmax(ensemble_scores)
        self.model = ensemble_models[best_idx]
        print(f"\nBest single model: Model {best_idx+1} with F1-score: {ensemble_scores[best_idx]:.4f}")

        # Create ensemble predictor
        self.ensemble_models = ensemble_models

        return self.model

    def predict_ensemble(self, X):
        """Make predictions using ensemble voting"""
        if hasattr(self, 'ensemble_models'):
            predictions = []
            for model in self.ensemble_models:
                predictions.append(model.predict_proba(X)[:, 1])

            # Average probabilities
            ensemble_probs = np.mean(predictions, axis=0)
            return (ensemble_probs >= 0.5).astype(int), ensemble_probs
        else:
            return self.model.predict(X), self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test, y_test):
        """Comprehensive model evaluation"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)

        # Get predictions
        if hasattr(self, 'ensemble_models'):
            y_pred, y_prob = self.predict_ensemble(X_test)
            print("Using ensemble predictions (5 models)")
        else:
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

        # Store metrics
        self.evaluation_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist()
        }

        return self.evaluation_metrics

    def save_model(self, filepath='improved_exoplanet_detector.pkl'):
        """Save the trained model and metadata"""
        print("\n" + "="*60)
        print("SAVING MODEL")
        print("="*60)

        model_package = {
            'model': self.model,
            'ensemble_models': self.ensemble_models if hasattr(self, 'ensemble_models') else None,
            'best_params': self.best_params,
            'evaluation_metrics': self.evaluation_metrics
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)

        print(f"  Model saved to {filepath}")

        # Save metrics as JSON
        metrics_filepath = filepath.replace('.pkl', '_metrics.json')
        with open(metrics_filepath, 'w') as f:
            json.dump({
                'best_params': self.best_params,
                'evaluation_metrics': self.evaluation_metrics,
                'ensemble_size': len(self.ensemble_models) if hasattr(self, 'ensemble_models') else 1
            }, f, indent=2)

        print(f"  Metrics saved to {metrics_filepath}")

def main():
    """Main training pipeline"""
    print("="*60)
    print("IMPROVED NASA EXOPLANET DETECTION MODEL TRAINING")
    print("="*60)

    # Initialize detector
    detector = ImprovedExoplanetDetector(random_state=42)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = detector.load_data()

    # Feature engineering
    X_train = detector.engineer_features(X_train)
    X_val = detector.engineer_features(X_val)
    X_test = detector.engineer_features(X_test)

    print(f"\nFeature engineering complete:")
    print(f"  Total features: {X_train.shape[1]}")

    # Hyperparameter optimization
    detector.optimize_hyperparameters(X_train, y_train, X_val, y_val)

    # Train ensemble for better performance
    detector.train_ensemble(X_train, y_train, X_val, y_val)

    # Evaluate
    detector.evaluate(X_test, y_test)

    # Save model
    detector.save_model('improved_exoplanet_detector.pkl')

    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nFinal Model Performance:")
    print(f"  F1-score: {detector.evaluation_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC: {detector.evaluation_metrics['roc_auc']:.4f}")

    if detector.evaluation_metrics['f1_score'] >= 0.95:
        print("\n[SUCCESS] Target of >95% F1-score achieved!")
    elif detector.evaluation_metrics['f1_score'] >= 0.90:
        print("\n[GOOD] F1-score >90% achieved. Close to target!")
    else:
        print(f"\n[INFO] Current F1-score: {detector.evaluation_metrics['f1_score']:.4f}")

    print("\nModel saved to: improved_exoplanet_detector.pkl")

if __name__ == "__main__":
    main()