"""
NASA Exoplanet Detection - Interactive Streamlit Application
============================================================
Interactive web app for detecting exoplanets using Kepler transit data.
Created for NASA Space Apps Challenge 2025.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="NASA Exoplanet Detection",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load both trained models and preprocessor"""
    models = {}

    # Load LightGBM model
    try:
        with open('exoplanet_detector.pkl', 'rb') as f:
            lgbm_package = pickle.load(f)
        with open('exoplanet_detector_metrics.json', 'r') as f:
            lgbm_metrics = json.load(f)
        models['LightGBM'] = {
            'package': lgbm_package,
            'metrics': lgbm_metrics
        }
    except Exception as e:
        st.warning(f"Could not load LightGBM model: {str(e)}")

    # Load TabNet model
    try:
        with open('../tabnet_exoplanet_detector.pkl', 'rb') as f:
            tabnet_package = pickle.load(f)
        with open('../tabnet_exoplanet_metrics.json', 'r') as f:
            tabnet_metrics = json.load(f)
        models['TabNet'] = {
            'package': tabnet_package,
            'metrics': tabnet_metrics
        }
    except Exception as e:
        st.warning(f"Could not load TabNet model: {str(e)}")

    # Load Random Forest model
    try:
        with open('../rf_exoplanet_detector.pkl', 'rb') as f:
            rf_package = pickle.load(f)
        with open('../rf_exoplanet_metrics.json', 'r') as f:
            rf_metrics = json.load(f)
        models['Random Forest'] = {
            'package': rf_package,
            'metrics': rf_metrics
        }
    except Exception as e:
        st.warning(f"Could not load Random Forest model: {str(e)}")

    # Load preprocessor
    try:
        with open('../data/preprocessing/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading preprocessor: {str(e)}")
        preprocessor = None

    return models, preprocessor

@st.cache_data
def load_sample_data():
    """Load sample test data for demonstration"""
    try:
        # Load scaled data for predictions
        X_test_scaled = pd.read_csv('../data/preprocessing/X_test.csv')
        # Load unscaled data for display
        X_test_unscaled = pd.read_csv('../data/preprocessing/X_test_unscaled.csv')
        y_test = pd.read_csv('../data/preprocessing/y_test.csv')
        return X_test_scaled, X_test_unscaled, y_test
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        return None, None, None

def create_feature_plot(features_df):
    """Create interactive feature distribution plot"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Orbital Period', 'Transit Depth', 'Planet Radius', 'Equilibrium Temperature')
    )

    features_to_plot = ['koi_period', 'koi_depth', 'koi_prad', 'koi_teq']
    positions = [(1,1), (1,2), (2,1), (2,2)]

    for feature, pos in zip(features_to_plot, positions):
        if feature in features_df.columns:
            fig.add_trace(
                go.Histogram(x=features_df[feature], name=feature, showlegend=False),
                row=pos[0], col=pos[1]
            )

    fig.update_layout(height=600, showlegend=False, title_text="Feature Distributions")
    return fig

def create_correlation_heatmap(features_df):
    """Create correlation heatmap"""
    corr_matrix = features_df.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title="Feature Correlation Matrix",
        height=600,
        xaxis_tickangle=-45
    )

    return fig

def predict_single(model, features, feature_names):
    """Make prediction for single instance"""
    # Create dataframe with proper column names
    input_df = pd.DataFrame([features], columns=feature_names)

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    return prediction, probability

def main():
    # Header
    st.markdown('<h1 class="main-header">🪐 NASA Exoplanet Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Detection of Exoplanets from Kepler Transit Data</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Load models and data
    models, preprocessor = load_models()
    X_test_scaled, X_test_unscaled, y_test = load_sample_data()

    if not models:
        st.error("Failed to load any models. Please ensure model files are in the correct location.")
        return

    if preprocessor is None:
        st.error("Failed to load preprocessor. Please ensure preprocessor file exists.")
        return

    feature_names = preprocessor['feature_columns']

    # Sidebar
    with st.sidebar:
        st.image("https://www.nasa.gov/wp-content/uploads/2023/03/nasa-logo-web-rgb.png", width=200)
        st.markdown("## NASA Space Apps Challenge 2025")
        st.markdown("**Challenge:** A World Away: Hunting for Exoplanets with AI")
        st.markdown("---")

        # Model selection
        st.markdown("### Model Selection")
        available_models = list(models.keys())
        selected_model_name = st.selectbox(
            "Choose Model:",
            available_models,
            help="LightGBM: Gradient boosting model\nTabNet: Neural network with attention\nRandom Forest: Ensemble of decision trees"
        )

        model_package = models[selected_model_name]['package']
        metrics = models[selected_model_name]['metrics']
        model = model_package['model']

        st.markdown(f"### {selected_model_name} Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("F1-Score", f"{metrics['evaluation_metrics']['f1_score']:.2%}")
            st.metric("Precision", f"{metrics['evaluation_metrics']['precision']:.2%}")
        with col2:
            st.metric("ROC-AUC", f"{metrics['evaluation_metrics']['roc_auc']:.2%}")
            st.metric("Recall", f"{metrics['evaluation_metrics']['recall']:.2%}")

        st.markdown("---")

        # Model comparison
        if len(models) > 1:
            st.markdown("### Model Comparison")
            comparison_df = pd.DataFrame({
                'Model': list(models.keys()),
                'F1-Score': [models[m]['metrics']['evaluation_metrics']['f1_score'] for m in models],
                'ROC-AUC': [models[m]['metrics']['evaluation_metrics']['roc_auc'] for m in models],
                'Accuracy': [models[m]['metrics']['evaluation_metrics']['accuracy'] for m in models]
            })
            comparison_df = comparison_df.set_index('Model')
            st.dataframe(comparison_df.style.format('{:.2%}'))

        st.markdown("---")
        st.markdown("### About")
        st.markdown(f"""
        This application uses **{selected_model_name}** to classify
        potential exoplanets from Kepler Space Telescope data.

        **Model Types:**
        - **LightGBM**: Gradient boosting ensemble
        - **TabNet**: Neural network with attention
        - **Random Forest**: Bootstrap aggregated trees

        **Features Used:**
        - Orbital Period, Transit Depth
        - Planet Radius, Equilibrium Temp
        - And 8 more stellar/orbital features
        """)

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predictions", "📊 Data Explorer", "📈 Model Performance", "📚 Documentation"])

    with tab1:
        st.header("Exoplanet Prediction Interface")

        prediction_method = st.radio(
            "Choose prediction method:",
            ["Manual Input", "Sample Data", "Upload CSV"]
        )

        if prediction_method == "Manual Input":
            st.subheader("Enter Transit Signal Parameters")

            col1, col2, col3 = st.columns(3)

            with col1:
                koi_period = st.number_input("Orbital Period (days)", min_value=0.0, max_value=1000.0, value=10.0, step=0.1)
                koi_depth = st.number_input("Transit Depth (ppm)", min_value=0.0, max_value=10000.0, value=500.0, step=10.0)
                koi_prad = st.number_input("Planet Radius (Earth radii)", min_value=0.0, max_value=50.0, value=2.0, step=0.1)
                koi_teq = st.number_input("Equilibrium Temperature (K)", min_value=0.0, max_value=3000.0, value=800.0, step=10.0)

            with col2:
                koi_impact = st.number_input("Impact Parameter", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
                koi_duration = st.number_input("Transit Duration (hours)", min_value=0.0, max_value=20.0, value=3.0, step=0.1)
                koi_model_snr = st.number_input("Model SNR", min_value=0.0, max_value=1000.0, value=30.0, step=1.0)
                koi_steff = st.number_input("Stellar Temperature (K)", min_value=0.0, max_value=10000.0, value=5800.0, step=100.0)

            with col3:
                koi_slogg = st.number_input("Stellar Surface Gravity", min_value=0.0, max_value=5.0, value=4.4, step=0.1)
                koi_srad = st.number_input("Stellar Radius (Solar radii)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
                koi_kepmag = st.number_input("Kepler Magnitude", min_value=0.0, max_value=20.0, value=14.0, step=0.1)
                koi_insol = st.number_input("Insolation Flux", min_value=0.0, max_value=5000.0, value=100.0, step=10.0)

            if st.button("🚀 Classify Signal", type="primary"):
                # Prepare features in correct order
                features = [koi_period, koi_depth, koi_prad, koi_teq, koi_impact,
                          koi_duration, koi_model_snr, koi_steff, koi_slogg,
                          koi_srad, koi_kepmag, koi_insol]

                # Scale features
                features_scaled = preprocessor['scaler'].transform([features])

                # Make prediction
                prediction = model.predict(features_scaled)[0]
                probability = model.predict_proba(features_scaled)[0]

                # Display results
                st.markdown("---")
                st.subheader("Classification Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if prediction == 1:
                        st.success("🌍 **PLANET DETECTED**")
                    else:
                        st.warning("❌ **FALSE POSITIVE**")

                with col2:
                    st.metric("Confidence", f"{max(probability):.2%}")

                with col3:
                    st.metric("Planet Probability", f"{probability[1]:.2%}")

                # Confidence gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability[1] * 100,
                    title={'text': "Planet Likelihood"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkgreen" if prediction == 1 else "darkred"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)

        elif prediction_method == "Sample Data":
            st.subheader("Test with Sample Data")

            if X_test_scaled is not None and X_test_unscaled is not None:
                sample_idx = st.slider("Select sample index", 0, len(X_test_scaled)-1, 0)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Sample Features:**")
                    # Use unscaled data for display
                    sample_features_display = X_test_unscaled.iloc[sample_idx]
                    for feat in ['koi_period', 'koi_depth', 'koi_prad', 'koi_teq']:
                        if feat in sample_features_display.index:
                            value = sample_features_display[feat]
                            # Format based on feature type
                            if feat == 'koi_period':
                                st.write(f"- Orbital Period: {value:.2f} days")
                            elif feat == 'koi_depth':
                                st.write(f"- Transit Depth: {value:.1f} ppm")
                            elif feat == 'koi_prad':
                                st.write(f"- Planet Radius: {value:.2f} Earth radii")
                            elif feat == 'koi_teq':
                                st.write(f"- Equilibrium Temp: {value:.0f} K")

                with col2:
                    if st.button("Classify Sample", type="primary"):
                        # Use scaled data for prediction
                        sample_features_scaled = X_test_scaled.iloc[sample_idx]
                        sample_input = sample_features_scaled.values.reshape(1, -1)
                        prediction = model.predict(sample_input)[0]
                        probability = model.predict_proba(sample_input)[0]
                        actual = y_test.iloc[sample_idx]['label']

                        st.markdown("**Results:**")
                        if prediction == 1:
                            st.success(f"Predicted: PLANET (confidence: {probability[1]:.2%})")
                        else:
                            st.warning(f"Predicted: FALSE POSITIVE (confidence: {probability[0]:.2%})")

                        if actual == 1:
                            st.info("Actual: PLANET")
                        else:
                            st.info("Actual: FALSE POSITIVE")

                        if prediction == actual:
                            st.success("✅ Correct prediction!")
                        else:
                            st.error("❌ Incorrect prediction")

        else:  # Upload CSV
            st.subheader("Upload Transit Data")
            uploaded_file = st.file_uploader(
                "Choose a CSV file with transit signal features",
                type="csv",
                help="File should contain columns: " + ", ".join(feature_names)
            )

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write(f"Loaded {len(df)} samples")

                    # Check for required columns
                    missing_cols = set(feature_names) - set(df.columns)
                    if missing_cols:
                        st.error(f"Missing columns: {missing_cols}")
                    else:
                        if st.button("Classify All Samples", type="primary"):
                            # Scale features
                            X_scaled = preprocessor['scaler'].transform(df[feature_names])

                            # Make predictions
                            predictions = model.predict(X_scaled)
                            probabilities = model.predict_proba(X_scaled)[:, 1]

                            # Add results to dataframe
                            df['Prediction'] = ['Planet' if p == 1 else 'False Positive' for p in predictions]
                            df['Planet_Probability'] = probabilities

                            # Display results
                            st.success(f"Classification complete!")
                            st.write(f"Planets detected: {sum(predictions)} / {len(predictions)}")

                            # Show results table
                            st.dataframe(df[['Prediction', 'Planet_Probability'] + feature_names[:4]])

                            # Download results
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name=f"exoplanet_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

    with tab2:
        st.header("Data Explorer")

        if X_test_unscaled is not None:
            st.subheader("Test Dataset Overview")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", len(X_test_unscaled))
            with col2:
                planet_count = sum(y_test['label'] == 1)
                st.metric("Confirmed Planets", planet_count)
            with col3:
                st.metric("False Positives", len(y_test) - planet_count)

            # Feature distributions
            st.subheader("Feature Distributions")
            fig_dist = create_feature_plot(X_test_unscaled)
            st.plotly_chart(fig_dist, use_container_width=True)

            # Correlation matrix
            st.subheader("Feature Correlations")
            fig_corr = create_correlation_heatmap(X_test_unscaled)
            st.plotly_chart(fig_corr, use_container_width=True)

            # Feature statistics
            st.subheader("Feature Statistics")
            # Display statistics with proper formatting
            stats = X_test_unscaled[['koi_period', 'koi_depth', 'koi_prad', 'koi_teq']].describe()
            stats_formatted = stats.copy()
            stats_formatted['koi_period'] = stats['koi_period'].map(lambda x: f"{x:.2f}")
            stats_formatted['koi_depth'] = stats['koi_depth'].map(lambda x: f"{x:.1f}")
            stats_formatted['koi_prad'] = stats['koi_prad'].map(lambda x: f"{x:.2f}")
            stats_formatted['koi_teq'] = stats['koi_teq'].map(lambda x: f"{x:.0f}")
            st.dataframe(stats_formatted)

    with tab3:
        st.header("Model Performance Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Performance Metrics")
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                'Score': [
                    metrics['evaluation_metrics']['accuracy'],
                    metrics['evaluation_metrics']['precision'],
                    metrics['evaluation_metrics']['recall'],
                    metrics['evaluation_metrics']['f1_score'],
                    metrics['evaluation_metrics']['roc_auc']
                ]
            })

            fig_metrics = px.bar(
                metrics_df, x='Metric', y='Score',
                title='Model Performance Metrics',
                color='Score',
                color_continuous_scale='viridis'
            )
            fig_metrics.add_hline(y=0.95, line_dash="dash", line_color="red",
                                annotation_text="Target: 95%")
            fig_metrics.update_layout(height=400)
            st.plotly_chart(fig_metrics, use_container_width=True)

        with col2:
            st.subheader("Confusion Matrix")
            cm = metrics['evaluation_metrics']['confusion_matrix']

            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['False Positive', 'Planet'],
                y=['False Positive', 'Planet'],
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 20},
                colorscale='Blues'
            ))
            fig_cm.update_layout(
                title="Prediction Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=400
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        # Feature importance
        st.subheader("Feature Importance")

        # Handle different model types
        importance_df = None
        if selected_model_name in ['LightGBM', 'Random Forest'] and hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
        elif selected_model_name == 'TabNet' and 'feature_importances' in metrics:
            importance_df = pd.DataFrame({
                'Feature': metrics['feature_importances']['features'],
                'Importance': metrics['feature_importances']['importances']
            }).sort_values('Importance', ascending=False)

        if importance_df is not None:
            fig_imp = px.bar(
                importance_df, x='Importance', y='Feature',
                orientation='h', title=f'{selected_model_name} Feature Importance Scores'
            )
            fig_imp.update_layout(height=500)
            st.plotly_chart(fig_imp, use_container_width=True)

        # Training history
        st.subheader("Training History")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Baseline Model**")
            st.write(f"- Train F1: {metrics['training_history']['baseline']['train_f1']:.4f}")
            st.write(f"- Val F1: {metrics['training_history']['baseline']['val_f1']:.4f}")
            st.write(f"- Training Time: {metrics['training_history']['baseline']['training_time']:.2f}s")

        with col2:
            st.markdown("**Tuned Model**")
            st.write(f"- Train F1: {metrics['training_history']['tuned']['train_f1']:.4f}")
            st.write(f"- Val F1: {metrics['training_history']['tuned']['val_f1']:.4f}")
            st.write(f"- Tuning Time: {metrics['training_history']['tuned']['tuning_time']:.2f}s")

    with tab4:
        st.header("Documentation")

        st.markdown("""
        ## About This Project

        This project was developed for the **NASA Space Apps Challenge 2025**, specifically for the
        "A World Away: Hunting for Exoplanets with AI" challenge. Our goal is to automate the detection
        of exoplanets from transit signals in the Kepler Space Telescope data.

        ### Dataset
        - **Source**: NASA Exoplanet Archive - Kepler Objects of Interest (KOI)
        - **Size**: 9,564 transit signals
        - **Classes**: Confirmed/Candidate planets vs False Positives
        - **Features**: 12 key features including orbital period, transit depth, planet radius, and stellar properties

        ### Model Architecture
        - **Algorithm**: LightGBM (Light Gradient Boosting Machine)
        - **Type**: Binary classification (Planet vs Non-Planet)
        - **Training**: 70% train, 10% validation, 20% test split
        - **Optimization**: Grid search hyperparameter tuning

        ### Key Features Used
        1. **koi_period**: Orbital period (days)
        2. **koi_depth**: Transit depth (parts per million)
        3. **koi_prad**: Planet radius (Earth radii)
        4. **koi_teq**: Equilibrium temperature (Kelvin)
        5. **koi_impact**: Impact parameter
        6. **koi_duration**: Transit duration (hours)
        7. **koi_model_snr**: Model signal-to-noise ratio
        8. **koi_steff**: Stellar effective temperature
        9. **koi_slogg**: Stellar surface gravity
        10. **koi_srad**: Stellar radius
        11. **koi_kepmag**: Kepler magnitude
        12. **koi_insol**: Insolation flux

        ### Performance
        - **F1-Score**: 83.16%
        - **ROC-AUC**: 92.56%
        - **Precision**: 82.72%
        - **Recall**: 83.60%

        ### How to Use
        1. **Manual Input**: Enter transit signal parameters manually
        2. **Sample Data**: Test with pre-loaded samples from the test set
        3. **Upload CSV**: Upload your own transit data for batch classification

        ### Team
        - Developed for NASA Space Apps Challenge 2025
        - Challenge: A World Away - Hunting for Exoplanets with AI

        ### References
        - NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/
        - Kepler Mission: https://www.nasa.gov/mission_pages/kepler/
        - LightGBM Documentation: https://lightgbm.readthedocs.io/

        ### Citation
        If you use this work, please cite:
        ```
        NASA Exoplanet Archive (2025)
        DOI: http://doi.org/10.17616/R3X31K
        ```
        """)

if __name__ == "__main__":
    main()