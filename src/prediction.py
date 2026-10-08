import pandas as pd
import numpy as np

class BearingPredictor:
    def __init__(self, model, health_analyzer, feature_extractor_func):
        self.model = model
        self.health_analyzer = health_analyzer
        self.extract_features = feature_extractor_func

    def predict_from_signal(self, signal, sampling_rate=20480):
        features = self.extract_features(signal, sampling_rate)
        health_score = self.health_analyzer.compute_health_score(features)
        condition, color, emoji = self.health_analyzer.classify_condition(health_score)
        alerts = self.health_analyzer.generate_alerts(features, health_score)
        
        result = {
            'features': features,
            'health_score': health_score,
            'condition': {
                'status': condition,
                'color': color,
                'emoji': emoji
            },
            'alerts': alerts
        }
        
        ml_pred = self.model.predict(features)
        if 'error' in ml_pred:
            result['ml_prediction'] = ml_pred['error']
        else:
            result['ml_prediction'] = ml_pred.get('prediction')
            result['confidence'] = ml_pred.get('confidence')
            result['probabilities'] = ml_pred.get('probabilities')
            
        return result

    def predict_from_file(self, filepath, test_name=None, channel=None):
        import pandas as pd
        df = pd.read_csv(filepath, sep='\t', header=None)
        
        if test_name == '1st_test':
            df = df.iloc[:, :8]
            df.columns = ['Bearing1_X', 'Bearing1_Y', 'Bearing2_X', 'Bearing2_Y', 'Bearing3_X', 'Bearing3_Y', 'Bearing4_X', 'Bearing4_Y']
        elif test_name in ['2nd_test', '4th_test']:
            df = df.iloc[:, :4]
            df.columns = ['Bearing1', 'Bearing2', 'Bearing3', 'Bearing4']
            
        return self.predict_from_dataframe(df, channel)

    def predict_from_dataframe(self, df, channel=None):
        results = {}
        cols = [channel] if channel and channel in df.columns else df.columns
        
        for col in cols:
            signal = df[col].values
            results[col] = self.predict_from_signal(signal)
            
        return results
