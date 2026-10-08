import numpy as np
import scipy.signal as signal

def bandpass_filter(data, lowcut, highcut, fs=20480, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y

def envelope_analysis(data, fs=20480):
    analytic_signal = signal.hilbert(data)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope

def compute_spectrogram(data, fs=20480, nperseg=1024):
    f, t, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg)
    return f, t, Sxx

def moving_average(data, window=100):
    weights = np.repeat(1.0, window) / window
    return np.convolve(data, weights, 'valid')

def normalize_signal(data):
    std = np.std(data)
    if std == 0: return data - np.mean(data)
    return (data - np.mean(data)) / std

def detect_anomalies_zscore(values, threshold=3.0):
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return np.zeros_like(values, dtype=bool)
    z_scores = np.abs((values - mean) / std)
    return z_scores > threshold
