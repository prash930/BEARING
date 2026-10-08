import pandas as pd
import numpy as np

class BearingHealthAnalyzer:
    def __init__(self):
        self.baseline = None
        self.weights = {
            'rms': 0.35,
            'kurtosis': 0.30,
            'crest_factor': 0.20,
            'peak': 0.15
        }

    def set_baseline(self, baseline_features):
        """Sets healthy baseline from early data."""
        self.baseline = baseline_features

    @staticmethod
    def detrend_signal(signal):
        """Remove DC component (mean) from signal for FFT analysis."""
        return signal - np.mean(signal)

    def compute_health_score(self, features_dict):
        """
        Health scores are computed from vibration feature deviations relative to an 
        early-life baseline, NOT from ground-truth labeled data.
        """
        if not self.baseline:
            return 100.0
        
        deviations = 0
        for feat, weight in self.weights.items():
            if feat in features_dict and feat in self.baseline:
                val = features_dict[feat]
                base_val = self.baseline[feat]
                ratio = (val - base_val) / base_val if base_val != 0 else 0
                deviations += weight * max(0, ratio) * 10
        
        score = max(0, 100.0 - deviations)
        return float(score)

    def classify_condition(self, health_score):
        if health_score >= 70:
            return ('HEALTHY', 'green', '🟢')
        elif health_score >= 40:
            return ('DEGRADING', 'orange', '🟡')
        else:
            return ('CRITICAL', 'red', '🔴')

    def compute_failure_probability(self, health_score):
        return max(0.0, min(100.0, 100.0 - health_score))

    def compute_degradation_trend(self, features_df, channel):
        df = features_df[features_df['channel'] == channel].copy()
        if df.empty:
            return df
            
        scores = []
        for _, row in df.iterrows():
            score = self.compute_health_score(row.to_dict())
            scores.append(score)
        
        df['health_score'] = scores
        if 'timestamp' in df.columns:
            return df.sort_values('timestamp')
        return df

    def generate_alerts(self, features_dict, health_score):
        alerts = []
        
        if not self.baseline:
            return alerts
            
        # RMS Alert
        if features_dict.get('rms', 0) > 3 * self.baseline.get('rms', 0):
            alerts.append({
                'level': 'WARNING',
                'message': 'RMS deviation is significantly high',
                'feature': 'rms',
                'value': features_dict.get('rms'),
                'threshold': 3 * self.baseline.get('rms', 0),
                'recommendation': 'Inspect bearing for signs of wear.'
            })
            
        # Kurtosis Alerts
        kurt = features_dict.get('kurtosis', 0)
        base_kurt = self.baseline.get('kurtosis', 0)
        if kurt > 3 * base_kurt or kurt > 10:
            alerts.append({
                'level': 'CRITICAL',
                'message': 'Kurtosis indicates severe impulsiveness',
                'feature': 'kurtosis',
                'value': kurt,
                'threshold': max(3 * base_kurt, 10),
                'recommendation': 'Immediate maintenance required.'
            })
        elif kurt > 2 * base_kurt or kurt > 6:
            alerts.append({
                'level': 'WARNING',
                'message': 'Kurtosis indicates moderate impulsiveness',
                'feature': 'kurtosis',
                'value': kurt,
                'threshold': max(2 * base_kurt, 6),
                'recommendation': 'Schedule maintenance check.'
            })
            
        # Crest Factor Alert
        cf = features_dict.get('crest_factor', 0)
        base_cf = self.baseline.get('crest_factor', 0)
        if cf > 2 * base_cf:
            alerts.append({
                'level': 'WARNING',
                'message': 'Crest factor is high',
                'feature': 'crest_factor',
                'value': cf,
                'threshold': 2 * base_cf,
                'recommendation': 'Monitor closely for defect progression.'
            })
            
        # Health Score Alerts
        if health_score < 40:
            alerts.append({
                'level': 'CRITICAL',
                'message': f'Overall health score is critical ({health_score:.1f})',
                'feature': 'health_score',
                'value': health_score,
                'threshold': 40,
                'recommendation': 'Stop equipment to prevent catastrophic failure.'
            })
        elif health_score < 70:
            alerts.append({
                'level': 'WARNING',
                'message': f'Overall health score is degrading ({health_score:.1f})',
                'feature': 'health_score',
                'value': health_score,
                'threshold': 70,
                'recommendation': 'Plan maintenance in upcoming cycle.'
            })
            
        return alerts
