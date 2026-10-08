import streamlit as st
import numpy as np
import numpy as np
import os
import sys
import plotly.graph_objects as go
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import IMSDataLoader, IMS_TEST_INFO
from src.feature_extraction import extract_features_batch
from src.health_analysis import BearingHealthAnalyzer

st.set_page_config(page_title='Bearing Analysis', page_icon='🔍', layout='wide')

def apply_custom_css():
    st.markdown("<style>div[data-testid='stMetric'] { background-color: #1e1e2e; padding: 1rem; border-radius: 10px; border: 1px solid #333; }</style>", unsafe_allow_html=True)

apply_custom_css()
st.title("🔍 Comprehensive Bearing Analysis")

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
loader = IMSDataLoader(data_dir)
tests = loader.get_available_tests()

if not tests:
    st.error("No test data found.")
    st.stop()

test_name = st.selectbox("Select Test", tests)
files = loader.get_file_list(test_name)

num_files = st.slider("Number of files to process", 5, min(500, len(files)), min(50, len(files)))

@st.cache_data(show_spinner="Extracting features...")
def process_data(t_name, n_files):
    flist = loader.get_file_list(t_name)
    if len(flist) > n_files:
        # Select evenly spaced files rather than random to preserve time structure
        indices = [int(i) for i in np.linspace(0, len(flist)-1, n_files).astype(int)]
        flist = [flist[i] for i in sorted(indices)]
    
    prog = st.progress(0)
    def cb(i, t): prog.progress(i/t)
    df_feat = extract_features_batch(flist, t_name, loader, progress_callback=cb)
    prog.empty()
    return df_feat


df_features = process_data(test_name, num_files)

tab1, tab2, tab3, tab4 = st.tabs(["Degradation Trend", "Feature Analysis", "Data Table", "Signal Details"])

with tab1:
    st.subheader("Bearing Degradation Trend")
    channel = st.selectbox("Channel", loader.get_channel_names(test_name), key='t1_ch')
    ch_df = df_features[df_features['channel'] == channel].copy()
    
    analyzer = BearingHealthAnalyzer()
    # Baseline = first file features (early, healthy operation)
    if len(ch_df) > 0:
        base = ch_df.iloc[0].to_dict()
        analyzer.set_baseline(base)
        
        scores = []
        for _, row in ch_df.iterrows():
            scores.append(analyzer.compute_health_score(row.to_dict()))
        ch_df['Health Score'] = scores
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ch_df['timestamp'], y=ch_df['Health Score'], mode='lines+markers', name='Health Score'))
        fig.add_hrect(y0=70, y1=100, fillcolor="green", opacity=0.2, line_width=0)
        fig.add_hrect(y0=40, y1=70, fillcolor="yellow", opacity=0.2, line_width=0)
        fig.add_hrect(y0=0, y1=40, fillcolor="red", opacity=0.2, line_width=0)
        fig.update_layout(template='plotly_dark', xaxis_title="Time", yaxis_title="Health Score", yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional trends
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ch_df['timestamp'], y=ch_df['rms'], name='RMS'))
        fig2.add_trace(go.Scatter(x=ch_df['timestamp'], y=ch_df['peak'], name='Peak'))
        fig2.update_layout(template='plotly_dark', title="RMS and Peak Trends")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Historical Feature Analysis")
    channel2 = st.selectbox("Channel", loader.get_channel_names(test_name), key='t2_ch')
    ch_df2 = df_features[df_features['channel'] == channel2]
    features_to_plot = st.multiselect("Select Features", ['rms', 'kurtosis', 'peak', 'crest_factor', 'dominant_frequency'], default=['rms', 'kurtosis'])
    fig3 = go.Figure()
    for f in features_to_plot:
        fig3.add_trace(go.Scatter(x=ch_df2['timestamp'], y=ch_df2[f], name=f))
    fig3.update_layout(template='plotly_dark')
    st.plotly_chart(fig3, use_container_width=True)
    st.download_button("Download CSV", data=ch_df2.to_csv(index=False), file_name="historical_features.csv")

with tab3:
    st.subheader("Data Table")
    st.dataframe(df_features)
    st.download_button("Download All Data", data=df_features.to_csv(index=False), file_name="all_features.csv")

with tab4:
    st.subheader("Signal Details")
    st.info("Select a file from the Data Table or Live Dashboard to view detailed waveform and FFTs here.")