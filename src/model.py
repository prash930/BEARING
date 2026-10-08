import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

class BearingModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.training_info = {}
        self.feature_columns = [
            'rms', 'std', 'peak', 'peak_to_peak', 'kurtosis', 'skewness', 
            'crest_factor', 'dominant_frequency', 'max_fft_amplitude', 
            'spectral_energy', 'band_energy_0_1k', 'band_energy_1k_5k', 
            'band_energy_5k_10k'
        ]

    def prepare_labels(self, features_df, test_name, data_loader):
        """
        Generate degradation-based labels using chronological position.
        Since IMS is run-to-failure, we use file position as a proxy for degradation stage.
        - First 70% of data per channel: Healthy
        - 70-90%: Degrading
        - Last 10%: Critical
        
        IMPORTANT: Labels are HEURISTIC and based on chronological position, 
        not ground-truth fault timing. Known failure bearings are identified 
        from IMS documentation.
        """
        test_info = data_loader.get_test_info().get(test_name, {})
        known_failures = test_info.get('known_failures', {})
        
        df = features_df.copy()
        df['label'] = 'Healthy'
        
        for channel in df['channel'].unique():
            mask = df['channel'] == channel
            channel_data = df[mask].sort_values('timestamp')
            n_samples = len(channel_data)
            
            if n_samples == 0:
                continue
            
            # Check if this channel corresponds to a known failed bearing
            bearing_id = channel.split('_')[0] if '_' in channel else channel
            is_failed_bearing = bearing_id in known_failures.keys()
            
            if is_failed_bearing and n_samples > 0:
                # For known failed bearings, use chronological degradation labeling
                idx_70 = int(0.7 * n_samples)
                idx_90 = int(0.9 * n_samples)
                
                # First 70%: Healthy, middle 20%: Degrading, last 10%: Critical
                healthy_indices = channel_data.iloc[:idx_70].index
                degrading_indices = channel_data.iloc[idx_70:idx_90].index
                critical_indices = channel_data.iloc[idx_90:].index
                
                df.loc[healthy_indices, 'label'] = 'Healthy'
                df.loc[degrading_indices, 'label'] = 'Degrading'
                df.loc[critical_indices, 'label'] = 'Critical'
            else:
                # For healthy bearings, label all as Healthy
                df.loc[mask, 'label'] = 'Healthy'
        
        return df

    def check_data_quality(self, features_df):
        """Check for NaN, Inf values and class balance before training."""
        issues = []
        
        # Check for NaN values
        nan_count = features_df[self.feature_columns].isna().sum().sum()
        if nan_count > 0:
            issues.append(f"WARNING: {nan_count} NaN values found in features")
        
        # Check for Inf values
        inf_count = np.isinf(features_df[self.feature_columns]).sum().sum()
        if inf_count > 0:
            issues.append(f"WARNING: {inf_count} Inf values found in features")
        
        # Check class balance if labels exist
        if 'label' in features_df.columns:
            label_counts = features_df['label'].value_counts()
            min_count = label_counts.min()
            max_count = label_counts.max()
            if min_count < 3:
                issues.append(f"WARNING: Minority class has only {min_count} samples (imbalanced)")
            print(f"Class distribution: {dict(label_counts)}")
        else:
            issues.append("WARNING: No 'label' column found in data")
        
        return issues

    def train(self, features_df, model_type='random_forest'):
        if 'label' not in features_df.columns:
            raise ValueError("DataFrame must contain a 'label' column for training.")
            
        # Check data quality
        quality_issues = self.check_data_quality(features_df)
        for issue in quality_issues:
            print(issue)
        
        # Display warning about heuristic labels
        print("\n" + "="*60)
        print("WARNING: Training labels are HEURISTIC (chronological position)")
        print("           NOT ground-truth fault labels.")
        print("           This is a degradation classification, not fault detection.")
        print("="*60 + "\n")
        
        X = features_df[self.feature_columns]
        y = self.label_encoder.fit_transform(features_df['label'])
        
        # Split by channel/time to avoid leakage - group by channel
        # Since consecutive samples are highly correlated, 
        # we split ensuring train and test have different channels
        unique_channels = features_df['channel'].unique()
        
        # Simple hold-out: use first 80% of channels for training, last 20% for testing
        n_train_channels = max(1, len(unique_channels) - 2)
        train_channels = unique_channels[:n_train_channels]
        test_channels = unique_channels[n_train_channels:]
        
        X_train = X[features_df['channel'].isin(train_channels)]
        X_test = X[features_df['channel'].isin(test_channels)]
        y_train = y[features_df['channel'].isin(train_channels)]
        y_test = y[features_df['channel'].isin(test_channels)]
        
        # If only one channel, use random split with stratification
        if len(unique_channels) <= 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
        else:
            # Scale using only training data
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            self.model = SVC(probability=True, random_state=42)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, target_names=self.label_encoder.classes_, zero_division=0),
            'train_channels': list(train_channels),
            'test_channels': list(test_channels),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        }
        
        if hasattr(self.model, 'feature_importances_'):
            metrics['feature_importance'] = dict(zip(self.feature_columns, self.model.feature_importances_))
        
        self.training_info = metrics
        return metrics

    def predict(self, features_dict):
        if not self.is_trained:
            return {'error': 'ML prediction unavailable - model not trained'}
        
        X = pd.DataFrame([features_dict])[self.feature_columns]
        # Handle NaN/Inf
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        X_scaled = self.scaler.transform(X)
        
        pred_idx = self.model.predict(X_scaled)[0]
        prediction = self.label_encoder.inverse_transform([pred_idx])[0]
        
        result = {
            'prediction': prediction,
            'classifier': 'Heuristic Degradation Classifier'
        }
        
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X_scaled)[0]
            result['confidence'] = float(np.max(probs))
            result['probabilities'] = dict(zip(self.label_encoder.classes_, probs))
        
        return result

    def save_model(self, model_dir):
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, 'bearing_model.pkl'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))

    def load_model(self, model_dir):
        self.model = joblib.load(os.path.join(model_dir, 'bearing_model.pkl'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
        self.is_trained = True

    def get_feature_importance(self):
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return {}
        return dict(zip(self.feature_columns, self.model.feature_importances_))