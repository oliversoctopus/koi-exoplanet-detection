"""
Data Preprocessing Pipeline for NASA Kepler Exoplanet Detection
================================================================
Prepares the Kepler KOI dataset for machine learning by handling
missing values, encoding labels, normalizing features, and splitting data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import pickle
import warnings
warnings.filterwarnings('ignore')

class ExoplanetPreprocessor:
    """Preprocessing pipeline for exoplanet detection"""

    def __init__(self, scaling_method='minmax'):
        """
        Initialize the preprocessor

        Parameters:
        -----------
        scaling_method : str
            Method for scaling features ('minmax' or 'standard')
        """
        self.scaling_method = scaling_method
        self.scaler = MinMaxScaler() if scaling_method == 'minmax' else StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = None
        self.label_mapping = {
            'FALSE POSITIVE': 0,
            'CANDIDATE': 1,
            'CONFIRMED': 1
        }

    def load_data(self, filepath):
        """Load the Kepler KOI dataset"""
        print("Loading dataset...")
        df = pd.read_csv(filepath, comment='#')
        print(f"  Loaded {df.shape[0]} samples with {df.shape[1]} features")
        return df

    def create_binary_labels(self, df):
        """Create binary labels from koi_disposition"""
        print("\nCreating binary labels...")

        # Map dispositions to binary labels
        df['label'] = df['koi_disposition'].map(self.label_mapping)

        # Check for any unmapped values
        unmapped = df[df['label'].isna()]['koi_disposition'].unique()
        if len(unmapped) > 0:
            print(f"  Warning: Unmapped dispositions found: {unmapped}")
            # Drop rows with unmapped dispositions
            df = df[df['label'].notna()]

        df['label'] = df['label'].astype(int)

        # Print label distribution
        label_counts = df['label'].value_counts()
        print(f"  Planets (1): {label_counts.get(1, 0)} samples ({label_counts.get(1, 0)/len(df)*100:.2f}%)")
        print(f"  Non-planets (0): {label_counts.get(0, 0)} samples ({label_counts.get(0, 0)/len(df)*100:.2f}%)")

        return df

    def select_features(self, df):
        """Select relevant features for model training"""
        print("\nSelecting features...")

        # Core features based on CLAUDE.md
        core_features = ['koi_period', 'koi_depth', 'koi_prad', 'koi_teq']

        # Additional potentially useful features
        additional_features = [
            'koi_impact', 'koi_duration', 'koi_model_snr',
            'koi_steff', 'koi_slogg', 'koi_srad',
            'koi_kepmag', 'koi_insol'
        ]

        # Combine features and check availability
        all_features = core_features + additional_features
        self.feature_columns = [f for f in all_features if f in df.columns]

        print(f"  Selected {len(self.feature_columns)} features:")
        print(f"  Core features: {[f for f in core_features if f in self.feature_columns]}")
        print(f"  Additional features: {[f for f in additional_features if f in self.feature_columns]}")

        # Check for features with too many missing values
        missing_threshold = 0.5  # Drop features with >50% missing
        features_to_keep = []

        for feature in self.feature_columns:
            missing_pct = df[feature].isnull().sum() / len(df)
            if missing_pct <= missing_threshold:
                features_to_keep.append(feature)
            else:
                print(f"  Dropping {feature} due to {missing_pct*100:.1f}% missing values")

        self.feature_columns = features_to_keep
        print(f"  Final feature count: {len(self.feature_columns)}")

        return df

    def handle_missing_values(self, X):
        """Handle missing values using median imputation"""
        print("\nHandling missing values...")

        # Check missing values before imputation
        missing_before = X.isnull().sum().sum()
        print(f"  Missing values before imputation: {missing_before}")

        # Apply imputation
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # Check missing values after imputation
        missing_after = X_imputed.isnull().sum().sum()
        print(f"  Missing values after imputation: {missing_after}")

        return X_imputed

    def scale_features(self, X_train, X_val, X_test):
        """Scale features using the specified method"""
        print(f"\nScaling features using {self.scaling_method}...")

        # Fit scaler on training data only
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        # Transform validation and test data
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )

        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        print(f"  Scaling complete. Feature ranges: [{X_train_scaled.min().min():.3f}, {X_train_scaled.max().max():.3f}]")

        return X_train_scaled, X_val_scaled, X_test_scaled

    def split_data(self, X, y, test_size=0.2, val_size=0.125, random_state=42):
        """
        Split data into train, validation, and test sets

        Parameters:
        -----------
        test_size : float
            Proportion for test set (default 0.2)
        val_size : float
            Proportion for validation set from training data (default 0.125, giving 70/10/20 split)
        """
        print("\nSplitting data...")

        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Second split: train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )

        print(f"  Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

        # Check class distribution in each set
        for name, y_set in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            planet_pct = (y_set == 1).sum() / len(y_set) * 100
            print(f"  {name} planet percentage: {planet_pct:.1f}%")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def preprocess(self, filepath, save_processed=True):
        """
        Full preprocessing pipeline

        Parameters:
        -----------
        filepath : str
            Path to the raw CSV file
        save_processed : bool
            Whether to save processed data to files

        Returns:
        --------
        Dictionary containing all preprocessed data and metadata
        """
        print("="*60)
        print("STARTING PREPROCESSING PIPELINE")
        print("="*60)

        # Load and prepare data
        df = self.load_data(filepath)
        df = self.create_binary_labels(df)
        df = self.select_features(df)

        # Separate features and labels
        X = df[self.feature_columns]
        y = df['label']

        # Handle missing values
        X = self.handle_missing_values(X)

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(X_train, X_val, X_test)

        # Create result dictionary
        processed_data = {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'label_mapping': self.label_mapping
        }

        if save_processed:
            self.save_processed_data(processed_data)

        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE!")
        print("="*60)

        return processed_data

    def save_processed_data(self, processed_data):
        """Save processed data to files"""
        print("\nSaving processed data...")

        # Save train/val/test sets as CSV
        processed_data['X_train'].to_csv('X_train.csv', index=False)
        processed_data['X_val'].to_csv('X_val.csv', index=False)
        processed_data['X_test'].to_csv('X_test.csv', index=False)

        processed_data['y_train'].to_csv('y_train.csv', index=False, header=['label'])
        processed_data['y_val'].to_csv('y_val.csv', index=False, header=['label'])
        processed_data['y_test'].to_csv('y_test.csv', index=False, header=['label'])

        print("  Saved train/val/test sets to CSV files")

        # Save preprocessor components for later use
        preprocessor_components = {
            'scaler': processed_data['scaler'],
            'imputer': processed_data['imputer'],
            'feature_columns': processed_data['feature_columns'],
            'label_mapping': processed_data['label_mapping']
        }

        with open('preprocessor.pkl', 'wb') as f:
            pickle.dump(preprocessor_components, f)

        print("  Saved preprocessor components to preprocessor.pkl")

        # Save a summary file
        summary = {
            'n_features': len(processed_data['feature_columns']),
            'feature_names': processed_data['feature_columns'],
            'n_train': len(processed_data['X_train']),
            'n_val': len(processed_data['X_val']),
            'n_test': len(processed_data['X_test']),
            'train_planet_pct': (processed_data['y_train'] == 1).sum() / len(processed_data['y_train']) * 100,
            'val_planet_pct': (processed_data['y_val'] == 1).sum() / len(processed_data['y_val']) * 100,
            'test_planet_pct': (processed_data['y_test'] == 1).sum() / len(processed_data['y_test']) * 100
        }

        with open('preprocessing_summary.txt', 'w') as f:
            f.write("PREPROCESSING SUMMARY\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Number of features: {summary['n_features']}\n")
            f.write(f"Features: {', '.join(summary['feature_names'])}\n\n")
            f.write("Dataset splits:\n")
            f.write(f"  Train: {summary['n_train']} samples ({summary['train_planet_pct']:.1f}% planets)\n")
            f.write(f"  Val: {summary['n_val']} samples ({summary['val_planet_pct']:.1f}% planets)\n")
            f.write(f"  Test: {summary['n_test']} samples ({summary['test_planet_pct']:.1f}% planets)\n")

        print("  Saved preprocessing summary to preprocessing_summary.txt")

def main():
    """Main execution function"""
    # Initialize preprocessor
    preprocessor = ExoplanetPreprocessor(scaling_method='minmax')

    # Run preprocessing pipeline
    processed_data = preprocessor.preprocess('../kepler_koi.csv', save_processed=True)

    # Display final summary
    print("\nFinal processed data shapes:")
    print(f"  X_train: {processed_data['X_train'].shape}")
    print(f"  X_val: {processed_data['X_val'].shape}")
    print(f"  X_test: {processed_data['X_test'].shape}")

    print("\nProcessed data saved to:")
    print("  - X_train.csv, X_val.csv, X_test.csv")
    print("  - y_train.csv, y_val.csv, y_test.csv")
    print("  - preprocessor.pkl (scaler, imputer, feature info)")
    print("  - preprocessing_summary.txt")

if __name__ == "__main__":
    main()