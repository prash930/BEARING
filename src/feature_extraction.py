import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.fft as fft

def extract_time_domain_features(signal):
    signal = np.asarray(signal)
    mean = np.mean(signal)
    std = np.std(signal)
    var = np.var(signal)
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    peak_to_peak = np.max(signal) - np.min(signal)
    # Fisher=True makes normal distribution = 0, add 3 back for actual kurtosis
    kurtosis = stats.kurtosis(signal, fisher=True) + 3.0
    skewness = stats.skew(signal)
    crest_factor = peak / rms if rms != 0 else 0
    
    return {
        'mean': mean,
        'std': std,
        'variance': var,
        'rms': rms,
        'peak': peak,
        'peak_to_peak': peak_to_peak,
        'kurtosis': kurtosis,
        'skewness': skewness,
        'crest_factor': crest_factor
    }

def compute_fft(signal, sampling_rate=20480):
    signal = np.asarray(signal)
    n = len(signal)
    amplitudes = np.abs(fft.rfft(signal)) / n
    frequencies = fft.rfftfreq(n, d=1.0/sampling_rate)
    return frequencies, amplitudes

def extract_frequency_domain_features(signal, sampling_rate=20480):
    frequencies, amplitudes = compute_fft(signal, sampling_rate)
    
    dominant_frequency = frequencies[np.argmax(amplitudes)]
    max_fft_amplitude = np.max(amplitudes)
    spectral_energy = np.sum(amplitudes**2)
    
    # Band energies
    band_0_1k = np.sum(amplitudes[(frequencies >= 0) & (frequencies < 1000)]**2)
    band_1k_5k = np.sum(amplitudes[(frequencies >= 1000) & (frequencies < 5000)]**2)
    band_5k_10k = np.sum(amplitudes[(frequencies >= 5000) & (frequencies < 10000)]**2)
    
    return {
        'dominant_frequency': dominant_frequency,
        'max_fft_amplitude': max_fft_amplitude,
        'spectral_energy': spectral_energy,
        'band_energy_0_1k': band_0_1k,
        'band_energy_1k_5k': band_1k_5k,
        'band_energy_5k_10k': band_5k_10k
    }

def extract_all_features(signal, sampling_rate=20480):
    features = extract_time_domain_features(signal)
    features.update(extract_frequency_domain_features(signal, sampling_rate))
    return features

def extract_features_from_dataframe(df, sampling_rate=20480):
    results = {}
    for col in df.columns:
        results[col] = extract_all_features(df[col].values, sampling_rate)
    return results

def extract_features_batch(file_list, test_name, data_loader, sampling_rate=20480, progress_callback=None):
    records = []
    total = len(file_list)
    import os
    
    for i, filepath in enumerate(file_list):
        filename = os.path.basename(filepath)
        timestamp = data_loader.parse_timestamp(filename)
        df = data_loader.load_file(filepath, test_name)
        
        features_dict = extract_features_from_dataframe(df, sampling_rate)
        for channel, features in features_dict.items():
            record = {
                'filename': filename,
                'timestamp': timestamp,
                'channel': channel
            }
            record.update(features)
            records.append(record)
            
        if progress_callback:
            progress_callback(i + 1, total)
            
    return pd.DataFrame(records)
