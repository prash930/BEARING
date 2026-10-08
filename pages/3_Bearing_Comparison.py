import streamlit as st
import os
import sys
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import IMSDataLoader, IMS_TEST_INFO
from src.feature_extraction import extract_time_domain_features, compute_fft

st.set_page_config(page_title='Bearing Comparison', page_icon='⚖️', layout='wide')
st.title("⚖️ Bearing Comparison")

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
loader = IMSDataLoader(data_dir)
tests = loader.get_available_tests()

if not tests:
    st.error("No test data found.")
    st.stop()

test_name = st.selectbox("Select Test", tests)
files = loader.get_file_list(test_name)
channels = loader.get_channel_names(test_name)

if not files:
    st.stop()

colA, colB = st.columns(2)
with colA:
    ref_idx = st.selectbox("Reference File", range(len(files)), format_func=lambda i: os.path.basename(files[i]), index=0)
with colB:
    test_idx = st.selectbox("Test File", range(len(files)), format_func=lambda i: os.path.basename(files[i]), index=max(0, len(files)-1))

channel = st.selectbox("Channel", channels)

ref_file = files[ref_idx]
test_file = files[test_idx]

ref_df = loader.load_file(ref_file, test_name)
test_df = loader.load_file(test_file, test_name)

ref_sig = ref_df[channel].values
test_sig = test_df[channel].values
fs = 20480

ref_feat = extract_time_domain_features(ref_sig)
test_feat = extract_time_domain_features(test_sig)

st.subheader("Waveform Comparison")
c1, c2 = st.columns(2)
with c1:
    fig_ref = go.Figure(go.Scatter(y=ref_sig[:2000]))
    fig_ref.update_layout(template='plotly_dark', title="Reference Waveform")
    st.plotly_chart(fig_ref, use_container_width=True)
with c2:
    fig_test = go.Figure(go.Scatter(y=test_sig[:2000]))
    fig_test.update_layout(template='plotly_dark', title="Test Waveform")
    st.plotly_chart(fig_test, use_container_width=True)

st.subheader("FFT Comparison")
c3, c4 = st.columns(2)
ref_f, ref_a = compute_fft(ref_sig, fs)
test_f, test_a = compute_fft(test_sig, fs)
with c3:
    f_ref = go.Figure(go.Scatter(x=ref_f, y=ref_a))
    f_ref.update_layout(template='plotly_dark', title="Reference FFT", xaxis_range=[0,5000])
    st.plotly_chart(f_ref, use_container_width=True)
with c4:
    f_test = go.Figure(go.Scatter(x=test_f, y=test_a))
    f_test.update_layout(template='plotly_dark', title="Test FFT", xaxis_range=[0,5000])
    st.plotly_chart(f_test, use_container_width=True)

st.subheader("Feature Comparison")
comp_data = []
for k in ref_feat:
    v_ref = ref_feat[k]
    v_test = test_feat.get(k, 0)
    change = v_test - v_ref
    pct = (change/v_ref*100) if v_ref != 0 else 0
    comp_data.append({"Feature": k, "Reference": v_ref, "Test": v_test, "Change": change, "Change %": pct})

import pandas as pd
st.dataframe(pd.DataFrame(comp_data).style.format({"Reference": "{:.4f}", "Test": "{:.4f}", "Change": "{:.4f}", "Change %": "{:.2f}%"}))

# Known failure info
st.markdown("---")
st.markdown("### 📋 Test Information")
if test_name in IMS_TEST_INFO:
    st.markdown(f"**Description:** {IMS_TEST_INFO[test_name].get('description', 'N/A')}")
    failures = IMS_TEST_INFO[test_name].get('known_failures', {})
    if failures:
        st.markdown("**Known Failures:**")
        for bearing, fault in failures.items():
            st.markdown(f"- **{bearing}:** {fault}")
    else:
        st.info("No documented failure information for this test.")
else:
    st.info("No documented failure information for this test.")

st.info("This comparison is based on vibration feature analysis. Percentage change shows how values differ between the two selected files.")