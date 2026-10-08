import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import IMSDataLoader, IMS_TEST_INFO
from src.feature_extraction import extract_features_batch, extract_frequency_domain_features
from src.model import BearingModel

st.set_page_config(page_title='Model Training', page_icon='🧠', layout='wide')
st.title("🧠 ML Model Training and Prediction")

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
loader = IMSDataLoader(data_dir)
tests = loader.get_available_tests()

if not tests:
    st.error("No test data found.")
    st.stop()

st.warning(
    "Labels are generated using a heuristic degradation-based approach (chronological position in run-to-failure test). "
    "These are NOT ground-truth fault labels. The IMS dataset is run-to-failure without explicit fault timing."
)

test_name = st.selectbox("Select Test for Training", tests)
channels = st.multiselect(
    "Select Channels", 
    loader.get_channel_names(test_name), 
    default=[loader.get_channel_names(test_name)[0]] if loader.get_channel_names(test_name) else []
)
model_type = st.selectbox("Model Type", ["random_forest", "svm", "gradient_boosting"])

if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None

FEATURE_COLUMNS = [
    'rms', 'std', 'peak', 'peak_to_peak', 'kurtosis', 'skewness', 
    'crest_factor', 'dominant_frequency', 'max_fft_amplitude', 
    'spectral_energy', 'band_energy_0_1k', 'band_energy_1k_5k', 
    'band_energy_5k_10k'
]

if st.button("Train Model"):
    model = BearingModel()
    
    with st.spinner("Extracting features..."):
        flist = loader.get_file_list(test_name)
        num_files = st.slider("Number of files to process", 5, min(500, len(flist)), min(50, len(flist)))
        if len(flist) > num_files:
            # Use evenly spaced files, not random, to preserve time structure
            import numpy as np
            indices = [int(i) for i in np.linspace(0, len(flist)-1, num_files).astype(int)]
            flist = [flist[i] for i in sorted(indices)]
        
        pbar = st.progress(0)
        def cb(i, t): pbar.progress(i/t)
        df_feat = extract_features_batch(flist, test_name, loader, progress_callback=cb)
        pbar.empty()
    
    # Filter selected channels
    df_feat = df_feat[df_feat['channel'].isin(channels)]
    
    if len(df_feat) == 0:
        st.error("No data selected for training. Please select at least one channel.")
    else:
        with st.spinner("Preparing labels and training..."):
            # Prepare labels (heuristic degradation-based)
            labeled_df = model.prepare_labels(df_feat, test_name, loader)
            
            # Check label distribution
            label_counts = labeled_df['label'].value_counts()
            st.info(f"Label distribution: {dict(label_counts)}")
            
            # Check for sufficient data in each class
            min_class_size = label_counts.min()
            if min_class_size < 3:
                st.warning(f"Warning: Minority class has only {min_class_size} samples. Model performance may be unreliable.")
            
            # Check for NaN/Inf values
            nan_count = labeled_df[FEATURE_COLUMNS].isna().sum().sum()
            if nan_count > 0:
                st.warning(f"Warning: {nan_count} NaN values found in features - will be handled during scaling.")
            
            # Train model with time-aware splitting (by channel, not random)
            try:
                metrics = model.train(labeled_df, model_type=model_type)
                
                st.session_state.trained_model = model
                st.success("Training Complete!")
                
                # Display metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{metrics.get('accuracy',0):.3f}")
                c2.metric("Precision", f"{metrics.get('precision',0):.3f}")
                c3.metric("Recall", f"{metrics.get('recall',0):.3f}")
                c4.metric("F1 Score", f"{metrics.get('f1',0):.3f}")
                
                # Show channel split info - this is key for time-series data
                st.caption(
                    f"Training channels: {', '.join(metrics.get('train_channels', []))} | "
                    f"Testing channels: {', '.join(metrics.get('test_channels', []))} | "
                    f"Train samples: {metrics.get('n_train_samples', 0)} | "
                    f"Test samples: {metrics.get('n_test_samples', 0)}"
                )
                
                # Feature importance
                importances = model.get_feature_importance()
                if importances:
                    fig = go.Figure(go.Bar(x=list(importances.keys()), y=list(importances.values())))
                    fig.update_layout(template='plotly_dark', title="Feature Importance")
                    st.plotly_chart(fig, use_container_width=True)
                    
            except ValueError as e:
                st.error(f"Training error: {e}")
                st.session_state.trained_model = None
        
        # Model Prediction section
st.subheader("Model Prediction")
if st.session_state.trained_model is None:
    st.info("ML prediction unavailable. Health assessment is based on vibration feature analysis only. Train a model first.")
else:
    st.info("Model is ready for prediction on new data.")
    # Add simple file selection for prediction
    test_file_pred = st.selectbox("Select file for prediction", loader.get_file_list(test_name)[:50])
    ch_pred = st.selectbox("Select Channel for prediction", channels)
    if st.button("Predict"):
        df_p = loader.load_file(test_file_pred, test_name)
        # Extract features using the same pipeline
        features_dict = extract_time_domain_features(df_p[ch_pred].values)
        # Add frequency features
        freq_feats = extract_frequency_domain_features(df_p[ch_pred].values, 20480)
        features_dict.update(freq_feats)
        
        res = st.session_state.trained_model.predict(features_dict)
        st.write("Prediction:", res.get('prediction', 'N/A'))
        if 'confidence' in res:
            st.write("Confidence:", f"{res['confidence']:.1f} %")
        if 'probabilities' in res:
            st.write("Probabilities:")
            for cls, prob in res['probabilities'].items():
                st.write(f"  {cls}: {prob:.1f}%")