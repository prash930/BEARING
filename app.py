import streamlit as st
from src.data_loader import IMSDataLoader

st.set_page_config(page_title="IMS Bearing Predictive Maintenance", layout="wide")

st.title("IMS Bearing Predictive Maintenance Dashboard")

loader = IMSDataLoader('.')
files = loader.get_file_list('1st_test')
content_dict = loader.get_test_metadata('1st_test')
content = {'files': files, 'test_name': content_dict.get('test_name', '1st_test'),
           'num_files': content_dict.get('num_files', 0),
           'channel_names': content_dict.get('channel_names', []),
           'test_info': content_dict}

st.write(f"Selected: **1st_test** with **{len(files)} files**")

st.write("---")
st.write("Use the navigation pages on the left for detailed analysis:")
st.write("- **Live Dashboard**: Vibration data, FFT, health score")
st.write("- **Bearing Analysis**: Feature extraction and analysis")
st.write("- **Bearing Comparison**: Compare bearing conditions")
st.write("- **Model Training**: Heuristic degradation classification")