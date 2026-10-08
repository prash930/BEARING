import streamlit as st
import os
import sys
import plotly.graph_objects as go
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import IMSDataLoader, IMS_TEST_INFO
from src.feature_extraction import extract_time_domain_features, compute_fft, extract_frequency_domain_features, extract_features_from_dataframe
from src.health_analysis import BearingHealthAnalyzer

st.set_page_config(page_title='Live Dashboard', page_icon='📈', layout='wide')

st.title("🔴 Live Monitoring Dashboard")

# Mode selector
mode = st.sidebar.selectbox(
    "🔧 Operating Mode",
    ["📊 Dataset Analysis", "🔴 Live MQTT (Future)"],
    key="live_dashboard_mode",
)

if mode == "🔴 Live MQTT (Future)":
    st.sidebar.info("MQTT mode requires ESP32 hardware connection. Currently in preview.")
    st.stop()

# Data loader
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
loader = IMSDataLoader(data_dir)
tests = loader.get_available_tests()

if not tests:
    st.error("No test data found.")
    st.stop()

test_name = st.sidebar.selectbox("📂 Select Test Dataset", tests)

if not test_name:
    st.warning("⬅️ Please select a test dataset from the sidebar.")
    st.stop()

# Load data with caching
@st.cache_data(show_spinner="Loading bearing data...")
def load_dashboard_data(_loader, _test_name):
    """Load first and last files for baseline and current comparison."""
    files = _loader.get_file_list(_test_name)
    if not files:
        return None

    first_df = _loader.load_file(files[0], _test_name)
    last_df = _loader.load_file(files[-1], _test_name)

    first_features = extract_features_from_dataframe(first_df)
    last_features = extract_features_from_dataframe(last_df)

    return {
        "first_features": first_features,
        "last_features": last_features,
        "first_file": os.path.basename(files[0]),
        "last_file": os.path.basename(files[-1]),
        "num_files": len(files),
        "files": files,
    }


from src.feature_extraction import extract_features_from_dataframe

with st.spinner("Loading bearing data..."):
    data = load_dashboard_data(loader, test_name)

if data is None:
    st.error("Could not load data for the selected test.")
    st.stop()

# Sidebar metadata
meta = loader.get_test_metadata(test_name)
with st.sidebar.expander("📋 Dataset Information", expanded=False):
    st.markdown(f"""
    - **Set:** {IMS_TEST_INFO[test_name]['set_number'] if test_name in IMS_TEST_INFO else 'N/A'}
    - **Files:** {meta['num_files']}
    - **Channels:** {meta['num_channels']}
    - **Sampling Rate:** {meta['sampling_rate']} Hz
    - **Samples/File:** {meta['samples_per_file']}
    - **Duration/File:** {meta['duration_seconds']} s
    - **Interval:** {meta['recording_interval']}
    """)
    st.markdown("**Known Failures:**")
    for bearing, fault in meta['known_failures'].items():
        st.markdown(f"- {bearing}: *{fault}*")

# Channel selector
channels = loader.get_channel_names(test_name)
channel = st.sidebar.selectbox("📡 Select Channel for Detailed View", channels)

# Health Analysis
analyzer = BearingHealthAnalyzer()
# Baseline = first file features (early, healthy operation)
analyzer.set_baseline(data["first_features"][channel])

current_features = data["last_features"][channel]
health_score = analyzer.compute_health_score(current_features)
prob_failure = analyzer.compute_failure_probability(health_score)
status, color, emoji = analyzer.classify_condition(health_score)

# Status indicator
st.markdown(f"""
<div style="padding:1rem; border-radius:10px; 
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-left: 6px solid {['#00ff41', '#ffaa00', '#ff0000'][['HEALTHY','DEGRADING','CRITICAL'].index(status)]};
            margin:1rem 0;">
    <h2 style="margin:0;">{emoji} {status}</h2>
    <p style="margin:0.3rem 0 0 0; opacity:0.8">Bearing Condition</p>
</div>
""", unsafe_allow_html=True)

# Health score explanation
baseline_features = data["first_features"][channel]
with st.expander("How is Health Score calculated?"):
    st.markdown("""
    Health Score is computed from vibration feature deviations relative to an early-life baseline.
    The calculation uses weighted deviations from the baseline:

    - **RMS** (35% weight): Root Mean Square of vibration amplitude
    - **Kurtosis** (30% weight): Measure of impulsiveness/peakedness
    - **Crest Factor** (20% weight): Peak/RMS ratio
    - **Peak** (15% weight): Maximum absolute amplitude

    Formula: 
    `deviations = Σ weight_i × max(0, (current_i - baseline_i) / baseline_i) × 10`
    `Health Score = max(0, 100 - deviations)`

    **Important**: The health score is an analytical indicator based on vibration feature 
    deviation from early-life baseline. It is NOT based on ground-truth fault labels.
    """)
    
    # Show baseline and current values
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption(f"Baseline RMS: {baseline_features.get('rms', 0):.4f}")
        st.caption(f"Current RMS: {current_features.get('rms', 0):.4f}")
    with col_b:
        st.caption(f"Baseline Kurtosis: {baseline_features.get('kurtosis', 0):.2f}")
        st.caption(f"Current Kurtosis: {current_features.get('kurtosis', 0):.2f}")

# ML classifier warning
st.markdown("---")
st.info(
    "⚠ **Heuristic Degradation Classifier**: The ML model uses chronological-based labels "
    "derived from file position, not ground-truth fault labels. Results show degradation "
    "progression classification only."
)

# Metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Health Score", f"{health_score:.1f} / 100")
with col2:
    st.metric("Current RMS", f"{current_features.get('rms', 0):.5f}")
with col3:
    st.metric("Current Kurtosis", f"{current_features.get('kurtosis', 0):.2f}")
with col4:
    st.metric("Crest Factor", f"{current_features.get('crest_factor', 0):.2f}")

col5, col6 = st.columns(2)
with col5:
    st.metric("Peak Vibration", f"{current_features.get('peak', 0):.5f}")
with col6:
    st.metric("Failure Probability", f"{prob_failure:.1f} %")

# Baseline vs Current comparison
st.markdown("---")
st.subheader("Baseline vs Current Comparison")

current_features = data["last_features"][channel]

c1, c2 = st.columns(2)
with c1:
    st.markdown("**BASELINE** (first file - early life)")
    st.caption(f"File: {data['first_file']}")
    st.markdown(f"- RMS: {baseline_features.get('rms', 0):.5f}")
    st.markdown(f"- Kurtosis: {baseline_features.get('kurtosis', 0):.2f}")
    st.markdown(f"- Peak: {baseline_features.get('peak', 0):.5f}")
    st.markdown(f"- Crest Factor: {baseline_features.get('crest_factor', 0):.2f}")

with c2:
    st.markdown("**CURRENT** (last file)")
    st.caption(f"File: {data['last_file']}")
    st.markdown(f"- RMS: {current_features.get('rms', 0):.5f}")
    st.markdown(f"- Kurtosis: {current_features.get('kurtosis', 0):.2f}")
    st.markdown(f"- Peak: {current_features.get('peak', 0):.5f}")
    st.markdown(f"- Crest Factor: {current_features.get('crest_factor', 0):.2f}")

# Change calculation
st.markdown("**CHANGE**")
change_data = []
for feat in ['rms', 'kurtosis', 'peak', 'crest_factor']:
    base = baseline_features.get(feat, 0)
    cur = current_features.get(feat, 0)
    change_pct = ((cur - base) / base * 100) if base != 0 else 0
    change_data.append(f"{feat}: {'+' if change_pct > 0 else ''}{change_pct:.1f}%")

for cd in change_data:
    st.caption(cd)

# Raw vibration graph
st.markdown("---")
st.subheader("Raw Bearing Vibration Signal")
fs = 20480

# Re-load signal data for waveform
first_df = loader.load_file(data["files"][0], test_name)
last_df = loader.load_file(data["files"][-1], test_name)
signal = last_df[channel].values

col_wave1, col_wave2 = st.columns(2)
with col_wave1:
    time_axis1 = np.arange(len(first_df[channel].values)) / fs
    fig_wave1 = go.Figure()
    fig_wave1.add_trace(go.Scatter(
        x=time_axis1, y=first_df[channel].values, 
        mode='lines', name=f"{channel} (First)"
    ))
    fig_wave1.update_layout(
        template='plotly_dark', 
        xaxis_title="Time (s)", 
        yaxis_title="Acceleration (g)",
        height=300
    )
    st.plotly_chart(fig_wave1, use_container_width=True)

with col_wave2:
    time_axis2 = np.arange(len(last_df[channel].values)) / fs
    fig_wave2 = go.Figure()
    fig_wave2.add_trace(go.Scatter(
        x=time_axis2, y=last_df[channel].values, 
        mode='lines', name=f"{channel} (Last)"
    ))
    fig_wave2.update_layout(
        template='plotly_dark', 
        xaxis_title="Time (s)", 
        yaxis_title="Acceleration (g)",
        height=300
    )
    st.plotly_chart(fig_wave2, use_container_width=True)

# FFT graph - exclude DC component (0 Hz) for dominant frequency
st.subheader("Bearing Vibration Frequency Spectrum")
signal_detrended = BearingHealthAnalyzer.detrend_signal(last_df[channel].values)
freqs, amps = compute_fft(signal_detrended, fs)

# Exclude 0 Hz bin from dominant frequency calculation
amps_no_dc = amps.copy()
amps_no_dc[0] = 0  # ignore DC component
dom_freq_idx = np.argmax(amps_no_dc)
dom_freq = freqs[dom_freq_idx]

fig_fft = go.Figure()
fig_fft.add_trace(go.Scatter(
    x=freqs, y=amps, mode='lines', name='Spectrum'
))
fig_fft.add_annotation(
    x=dom_freq, y=amps[dom_freq_idx], 
    text=f"Dom: {dom_freq:.1f}Hz (DC excluded)", showarrow=True
)
fig_fft.update_layout(
    template='plotly_dark', 
    xaxis_title="Frequency (Hz)", 
    yaxis_title="Amplitude", 
    xaxis_range=[0, 5000],
    height=400
)
st.plotly_chart(fig_fft, use_container_width=True)

# Health trend information
st.markdown("---")
st.subheader("Health Analysis Overview")

# Quick channel overview
st.caption(f"Comparing **first file** ({data['first_file']}) vs **last file** ({data['last_file']})")

n_cols = min(len(channels), 6)
cols = st.columns(n_cols)
for i, ch in enumerate(channels):
    with cols[i % n_cols]:
        analyzer_ch = BearingHealthAnalyzer()
        analyzer_ch.set_baseline(data["first_features"][ch])
        h_score = analyzer_ch.compute_health_score(data["last_features"][ch])
        s, c, e = analyzer_ch.classify_condition(h_score)

        bg_colors = {"HEALTHY": "#1a5c1a", "DEGRADING": "#5c4500", "CRITICAL": "#5c1a1a"}
        border_colors = {"HEALTHY": "#00ff41", "DEGRADING": "#ffaa00", "CRITICAL": "#ff0000"}
        st.markdown(
            f'<div style="padding:8px; text-align-center; '
            f'background-color:{bg_colors[s]}; border-radius:6px; '
            f'border-left:4px solid {border_colors[s]}; color:white; margin-bottom:4px;">'
            f'<b>{ch}</b><br>{e} {s}<br>'
            f'<span style="font-size:1rem; font-weight:500">{h_score:.0f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# Alerts
alerts = analyzer.generate_alerts(current_features, health_score)
if alerts:
    st.markdown("---")
    st.subheader("⚠️ Active Alerts")
    for alert in alerts:
        level = alert.get("level", "WARNING")
        css_cls = "CRITICAL" if level == "CRITICAL" else "WARNING"
        icon = "🔴" if level == "CRITICAL" else "🟡"
        st.markdown(
            f'<div style="padding:8px; background-color:#2a1a1a; '
            f'border-left:4px solid #ff0000; border-radius:6px; '
            f'margin:4px 0; color:#ffd;">'
            f'<b>{icon} {level}</b>: {alert["message"]}<br>'
            f'Feature: <code>{alert["feature"]}</code> = {alert["value"]:.4f} '
            f'(threshold: {alert["threshold"]:.4f})<br>'
            f'<i>Recommendation: {alert["recommendation"]}</i>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")
st.caption(
    "AI-Based Bearing Predictive Maintenance System · "
    "NASA IMS Bearing Dataset · "
    f"Test: {test_name} · Files: {data['num_files']}"
)