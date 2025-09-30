"""
Data Exploration Script for NASA Kepler Exoplanet Detection
============================================================
Explores the Kepler Objects of Interest (KOI) dataset to understand
data structure, distributions, and quality for exoplanet classification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data(filepath):
    """Load the Kepler KOI dataset"""
    print("Loading Kepler KOI dataset...")
    # Skip comment lines at the beginning of the file
    df = pd.read_csv(filepath, comment='#')
    print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def basic_info(df):
    """Display basic information about the dataset"""
    print("\n" + "="*60)
    print("DATASET OVERVIEW")
    print("="*60)

    print(f"\nDataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\nColumn names and types:")
    print("-"*40)
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print("-"*40)
    print(df.head())

    print("\nBasic statistics:")
    print("-"*40)
    print(df.describe())

    return df

def analyze_target_variable(df):
    """Analyze the target variable (koi_disposition)"""
    print("\n" + "="*60)
    print("TARGET VARIABLE ANALYSIS")
    print("="*60)

    if 'koi_disposition' in df.columns:
        print("\nTarget variable distribution (koi_disposition):")
        print("-"*40)
        disposition_counts = df['koi_disposition'].value_counts()
        print(disposition_counts)
        print("\nPercentages:")
        print(disposition_counts / len(df) * 100)

        # Create binary label
        df['is_planet'] = df['koi_disposition'].isin(['CONFIRMED', 'CANDIDATE']).astype(int)
        binary_counts = df['is_planet'].value_counts()
        print("\nBinary classification distribution:")
        print(f"Planets (CONFIRMED/CANDIDATE): {binary_counts.get(1, 0)} ({binary_counts.get(1, 0)/len(df)*100:.2f}%)")
        print(f"Non-planets (FALSE POSITIVE): {binary_counts.get(0, 0)} ({binary_counts.get(0, 0)/len(df)*100:.2f}%)")

        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        disposition_counts.plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
        axes[0].set_title('KOI Disposition Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Disposition')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)

        binary_counts.plot(kind='pie', ax=axes[1], autopct='%1.1f%%',
                          labels=['Non-Planet', 'Planet'],
                          colors=['coral', 'lightgreen'])
        axes[1].set_title('Binary Classification Distribution', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('')

        plt.tight_layout()
        plt.savefig('../preprocessing/target_distribution.png', dpi=100)
        plt.show()

    else:
        print("Warning: 'koi_disposition' column not found in dataset")

    return df

def analyze_key_features(df):
    """Analyze key features for exoplanet detection"""
    print("\n" + "="*60)
    print("KEY FEATURES ANALYSIS")
    print("="*60)

    key_features = ['koi_period', 'koi_depth', 'koi_prad', 'koi_teq', 'koi_score']
    available_features = [f for f in key_features if f in df.columns]

    print(f"\nAvailable key features: {available_features}")

    if available_features:
        feature_stats = df[available_features].describe()
        print("\nFeature statistics:")
        print("-"*40)
        print(feature_stats)

        # Visualize distributions
        n_features = len(available_features)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, feature in enumerate(available_features):
            if idx < 6:
                axes[idx].hist(df[feature].dropna(), bins=50, edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
                axes[idx].set_xlabel(feature)
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(available_features), 6):
            axes[idx].axis('off')

        plt.suptitle('Key Feature Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('../preprocessing/feature_distributions.png', dpi=100)
        plt.show()

    return df

def analyze_missing_values(df):
    """Analyze missing values in the dataset"""
    print("\n" + "="*60)
    print("MISSING VALUES ANALYSIS")
    print("="*60)

    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100

    missing_summary = pd.DataFrame({
        'Missing_Count': missing_counts,
        'Missing_Percentage': missing_percentages
    })
    missing_summary = missing_summary[missing_summary['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

    if len(missing_summary) > 0:
        print("\nColumns with missing values:")
        print("-"*40)
        print(missing_summary)

        # Visualize missing values
        if len(missing_summary) > 0:
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(missing_summary)), missing_summary['Missing_Percentage'],
                   color='salmon', edgecolor='black')
            plt.xticks(range(len(missing_summary)), missing_summary.index, rotation=45, ha='right')
            plt.xlabel('Features')
            plt.ylabel('Missing Percentage (%)')
            plt.title('Missing Values by Feature', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('../preprocessing/missing_values.png', dpi=100)
            plt.show()
    else:
        print("\nNo missing values found in the dataset!")

    return df

def analyze_correlations(df):
    """Analyze correlations between numerical features"""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)

    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if 'is_planet' in numerical_cols:
        # Focus on features correlated with target
        if len(numerical_cols) > 1:
            correlations_with_target = df[numerical_cols].corr()['is_planet'].sort_values(ascending=False)
            print("\nCorrelations with target variable (is_planet):")
            print("-"*40)
            print(correlations_with_target)

            # Create correlation heatmap for top features
            key_features = ['koi_period', 'koi_depth', 'koi_prad', 'koi_teq', 'koi_score', 'is_planet']
            available_features = [f for f in key_features if f in numerical_cols]

            if len(available_features) > 2:
                plt.figure(figsize=(10, 8))
                correlation_matrix = df[available_features].corr()
                sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                           center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
                plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig('../preprocessing/correlation_matrix.png', dpi=100)
                plt.show()

    return df

def analyze_by_disposition(df):
    """Analyze features by disposition (planet vs non-planet)"""
    print("\n" + "="*60)
    print("FEATURE ANALYSIS BY DISPOSITION")
    print("="*60)

    if 'is_planet' in df.columns:
        key_features = ['koi_period', 'koi_depth', 'koi_prad', 'koi_teq']
        available_features = [f for f in key_features if f in df.columns]

        if available_features:
            # Statistical comparison
            print("\nFeature statistics by disposition:")
            print("-"*40)
            for feature in available_features:
                planet_values = df[df['is_planet']==1][feature].dropna()
                nonplanet_values = df[df['is_planet']==0][feature].dropna()

                print(f"\n{feature}:")
                print(f"  Planets - Mean: {planet_values.mean():.4f}, Median: {planet_values.median():.4f}")
                print(f"  Non-planets - Mean: {nonplanet_values.mean():.4f}, Median: {nonplanet_values.median():.4f}")

            # Visualize distributions by class
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()

            for idx, feature in enumerate(available_features[:4]):
                planet_data = df[df['is_planet']==1][feature].dropna()
                nonplanet_data = df[df['is_planet']==0][feature].dropna()

                axes[idx].hist(planet_data, bins=30, alpha=0.5, label='Planet', color='green', edgecolor='black')
                axes[idx].hist(nonplanet_data, bins=30, alpha=0.5, label='Non-Planet', color='red', edgecolor='black')
                axes[idx].set_title(f'{feature} Distribution by Class', fontsize=12, fontweight='bold')
                axes[idx].set_xlabel(feature)
                axes[idx].set_ylabel('Frequency')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)

            plt.suptitle('Feature Distributions by Planet Classification', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('../preprocessing/features_by_class.png', dpi=100)
            plt.show()

    return df

def generate_summary_report(df):
    """Generate a summary report of the exploration"""
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)

    print("\n[Dataset Summary]")
    print(f"  - Total samples: {len(df):,}")
    print(f"  - Total features: {df.shape[1]}")
    print(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    if 'is_planet' in df.columns:
        planet_count = df['is_planet'].sum()
        print(f"\n[Target Distribution]")
        print(f"  - Planets: {planet_count:,} ({planet_count/len(df)*100:.2f}%)")
        print(f"  - Non-planets: {len(df)-planet_count:,} ({(len(df)-planet_count)/len(df)*100:.2f}%)")
        print(f"  - Class imbalance ratio: 1:{(len(df)-planet_count)/planet_count:.2f}")

    missing_counts = df.isnull().sum()
    cols_with_missing = (missing_counts > 0).sum()
    print(f"\n[Data Quality]")
    print(f"  - Columns with missing values: {cols_with_missing}")
    print(f"  - Total missing values: {missing_counts.sum():,}")
    print(f"  - Missing data percentage: {missing_counts.sum()/(df.shape[0]*df.shape[1])*100:.2f}%")

    print("\n[Key Insights]")
    print("  - Dataset shows moderate class imbalance (~50% false positives)")
    print("  - Key features include orbital period, transit depth, planet radius, and temperature")
    print("  - Missing values present in some columns - imputation strategy needed")
    print("  - Features show different distributions between planets and non-planets")

    print("\n[Next Steps]")
    print("  1. Handle missing values through imputation or removal")
    print("  2. Normalize/scale numerical features")
    print("  3. Encode categorical variables if needed")
    print("  4. Split data into train/validation/test sets")
    print("  5. Train LightGBM classifier with class weight adjustment")

if __name__ == "__main__":
    # Load the dataset
    df = load_data('../kepler_koi.csv')

    # Perform comprehensive exploration
    df = basic_info(df)
    df = analyze_target_variable(df)
    df = analyze_key_features(df)
    df = analyze_missing_values(df)
    df = analyze_correlations(df)
    df = analyze_by_disposition(df)

    # Generate summary report
    generate_summary_report(df)

    print("\n" + "="*60)
    print("Data exploration completed successfully!")
    print("Visualizations saved in the preprocessing folder.")
    print("="*60)