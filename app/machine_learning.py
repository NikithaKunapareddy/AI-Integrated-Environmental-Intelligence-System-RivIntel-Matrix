"""
Advanced Machine Learning Module for River Intelligence System
Provides comprehensive ML algorithms for environmental data analysis
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import joblib
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class WaterQualityPredictor:
    """
    Advanced water quality prediction system using ensemble methods
    """
    
    def __init__(self):
        self.models = {
            'rf_regressor': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb_regressor': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svr': SVR(kernel='rbf'),
            'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.trained_models = {}
        self.feature_importance = {}
        
    def preprocess_data(self, data):
        """
        Comprehensive data preprocessing for water quality analysis
        """
        # Handle missing values
        data = data.fillna(data.mean() if data.select_dtypes(include=[np.number]).shape[1] > 0 else data.mode().iloc[0])
        
        # Feature engineering
        if 'timestamp' in data.columns:
            data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
            data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
            data['month'] = pd.to_datetime(data['timestamp']).dt.month
            data['season'] = data['month'].apply(self._get_season)
        
        # Create interaction features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    data[f'{col1}_x_{col2}'] = data[col1] * data[col2]
                    data[f'{col1}_ratio_{col2}'] = data[col1] / (data[col2] + 1e-8)
        
        return data
    
    def _get_season(self, month):
        """Convert month to season"""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall
    
    def train_models(self, X, y, test_size=0.2):
        """
        Train multiple models and select best performing one
        """
        X_processed = self.preprocess_data(X)
        X_scaled = self.scaler.fit_transform(X_processed)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        best_score = -np.inf
        best_model = None
        
        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                
                if score > best_score:
                    best_score = score
                    best_model = name
                
                self.trained_models[name] = {
                    'model': model,
                    'score': score,
                    'predictions': model.predict(X_test),
                    'mse': mean_squared_error(y_test, model.predict(X_test))
                }
                
                logging.info(f"Model {name} trained with score: {score:.4f}")
                
            except Exception as e:
                logging.error(f"Error training model {name}: {str(e)}")
        
        return best_model, best_score
    
    def predict_water_quality(self, features, model_name=None):
        """
        Predict water quality using trained models
        """
        if not self.trained_models:
            raise ValueError("No trained models available. Please train models first.")
        
        features_processed = self.preprocess_data(pd.DataFrame([features]))
        features_scaled = self.scaler.transform(features_processed)
        
        if model_name and model_name in self.trained_models:
            model = self.trained_models[model_name]['model']
            return model.predict(features_scaled)[0]
        else:
            # Ensemble prediction
            predictions = []
            weights = []
            
            for name, model_info in self.trained_models.items():
                pred = model_info['model'].predict(features_scaled)[0]
                predictions.append(pred)
                weights.append(model_info['score'])
            
            weighted_prediction = np.average(predictions, weights=weights)
            return weighted_prediction

class PollutionDetector:
    """
    Advanced pollution detection and classification system
    """
    
    def __init__(self):
        self.pollution_types = [
            'chemical', 'biological', 'physical', 'thermal', 'radioactive'
        ]
        self.severity_levels = ['low', 'medium', 'high', 'critical']
        self.classifier = GradientBoostingClassifier(n_estimators=200, random_state=42)
        self.severity_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def generate_pollution_features(self, data):
        """
        Extract pollution-specific features from environmental data
        """
        features = {}
        
        # Chemical pollution indicators
        if 'ph' in data:
            features['ph_deviation'] = abs(data['ph'] - 7.0)
            features['ph_acidic'] = 1 if data['ph'] < 6.5 else 0
            features['ph_alkaline'] = 1 if data['ph'] > 8.5 else 0
        
        # Biological pollution indicators
        if 'dissolved_oxygen' in data:
            features['low_oxygen'] = 1 if data['dissolved_oxygen'] < 5.0 else 0
            features['oxygen_saturation'] = data.get('dissolved_oxygen', 0) / 14.6  # Normalize
        
        # Physical pollution indicators
        if 'turbidity' in data:
            features['high_turbidity'] = 1 if data['turbidity'] > 10 else 0
            features['turbidity_level'] = min(data['turbidity'] / 100, 1.0)
        
        # Temperature-based features
        if 'temperature' in data:
            features['temp_anomaly'] = abs(data['temperature'] - 20) / 20  # Normalize around 20°C
            features['thermal_pollution'] = 1 if data['temperature'] > 30 else 0
        
        # Conductivity features
        if 'conductivity' in data:
            features['high_conductivity'] = 1 if data['conductivity'] > 1000 else 0
            features['conductivity_norm'] = min(data['conductivity'] / 2000, 1.0)
        
        return features
    
    def train_pollution_detector(self, training_data):
        """
        Train pollution detection models
        """
        X = []
        y_type = []
        y_severity = []
        
        for record in training_data:
            features = self.generate_pollution_features(record['data'])
            X.append(list(features.values()))
            y_type.append(record['pollution_type'])
            y_severity.append(record['severity'])
        
        X = np.array(X)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train pollution type classifier
        self.classifier.fit(X_scaled, y_type)
        
        # Train severity classifier
        self.severity_classifier.fit(X_scaled, y_severity)
        
        self.is_trained = True
        
        # Evaluate models
        type_score = cross_val_score(self.classifier, X_scaled, y_type, cv=5).mean()
        severity_score = cross_val_score(self.severity_classifier, X_scaled, y_severity, cv=5).mean()
        
        logging.info(f"Pollution type classifier accuracy: {type_score:.4f}")
        logging.info(f"Severity classifier accuracy: {severity_score:.4f}")
        
        return type_score, severity_score
    
    def detect_pollution(self, environmental_data):
        """
        Detect pollution type and severity from environmental data
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Please train the model first.")
        
        features = self.generate_pollution_features(environmental_data)
        X = np.array([list(features.values())])
        X_scaled = self.scaler.transform(X)
        
        # Predict pollution type and probability
        pollution_type = self.classifier.predict(X_scaled)[0]
        type_probabilities = self.classifier.predict_proba(X_scaled)[0]
        
        # Predict severity
        severity = self.severity_classifier.predict(X_scaled)[0]
        severity_probabilities = self.severity_classifier.predict_proba(X_scaled)[0]
        
        return {
            'pollution_detected': True if max(type_probabilities) > 0.6 else False,
            'pollution_type': pollution_type,
            'type_confidence': max(type_probabilities),
            'severity': severity,
            'severity_confidence': max(severity_probabilities),
            'features_analyzed': features,
            'risk_score': self._calculate_risk_score(features, max(type_probabilities), max(severity_probabilities))
        }
    
    def _calculate_risk_score(self, features, type_confidence, severity_confidence):
        """
        Calculate overall environmental risk score
        """
        base_score = (type_confidence + severity_confidence) / 2
        
        # Adjust based on critical features
        risk_multipliers = 0
        if features.get('ph_deviation', 0) > 2:
            risk_multipliers += 0.2
        if features.get('low_oxygen', 0) == 1:
            risk_multipliers += 0.3
        if features.get('high_turbidity', 0) == 1:
            risk_multipliers += 0.1
        if features.get('thermal_pollution', 0) == 1:
            risk_multipliers += 0.15
        
        final_score = min(base_score * (1 + risk_multipliers), 1.0)
        return final_score

class EnvironmentalForecaster:
    """
    Time series forecasting for environmental parameters
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.forecast_horizons = [1, 7, 30]  # 1 day, 1 week, 1 month
        
    def prepare_time_series_data(self, data, target_column, sequence_length=10):
        """
        Prepare time series data for LSTM-style prediction
        """
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i+sequence_length])
            targets.append(data[i+sequence_length])
        
        return np.array(sequences), np.array(targets)
    
    def train_forecasting_model(self, historical_data, target_parameter):
        """
        Train forecasting model for specific environmental parameter
        """
        # Simple moving average and linear trend model
        data_series = historical_data[target_parameter].values
        
        # Calculate moving averages
        ma_7 = pd.Series(data_series).rolling(window=7).mean()
        ma_30 = pd.Series(data_series).rolling(window=30).mean()
        
        # Calculate trend
        x = np.arange(len(data_series))
        trend_model = LinearRegression()
        trend_model.fit(x.reshape(-1, 1), data_series)
        
        # Store models
        self.models[target_parameter] = {
            'trend_model': trend_model,
            'ma_7': ma_7.iloc[-1] if not pd.isna(ma_7.iloc[-1]) else np.mean(data_series),
            'ma_30': ma_30.iloc[-1] if not pd.isna(ma_30.iloc[-1]) else np.mean(data_series),
            'last_values': data_series[-10:],
            'mean': np.mean(data_series),
            'std': np.std(data_series)
        }
        
        return True
    
    def forecast_parameter(self, parameter, steps_ahead=7):
        """
        Forecast environmental parameter for specified steps ahead
        """
        if parameter not in self.models:
            raise ValueError(f"No trained model for parameter: {parameter}")
        
        model_info = self.models[parameter]
        
        # Trend-based prediction
        last_index = len(model_info['last_values'])
        future_indices = np.arange(last_index, last_index + steps_ahead).reshape(-1, 1)
        trend_predictions = model_info['trend_model'].predict(future_indices)
        
        # Moving average influence
        ma_influence = (model_info['ma_7'] + model_info['ma_30']) / 2
        
        # Combine predictions with uncertainty
        predictions = []
        confidence_intervals = []
        
        for i, trend_pred in enumerate(trend_predictions):
            # Weight recent moving average more heavily for near-term predictions
            weight = max(0.1, 1 - (i * 0.1))  # Decreasing weight for longer horizons
            prediction = trend_pred * (1 - weight) + ma_influence * weight
            
            # Calculate confidence interval
            uncertainty = model_info['std'] * (1 + i * 0.1)  # Increasing uncertainty
            conf_lower = prediction - 1.96 * uncertainty
            conf_upper = prediction + 1.96 * uncertainty
            
            predictions.append(prediction)
            confidence_intervals.append((conf_lower, conf_upper))
        
        return {
            'predictions': predictions,
            'confidence_intervals': confidence_intervals,
            'forecast_dates': [datetime.now() + timedelta(days=i+1) for i in range(steps_ahead)],
            'parameter': parameter,
            'model_accuracy': self._estimate_model_accuracy(model_info)
        }
    
    def _estimate_model_accuracy(self, model_info):
        """
        Estimate model accuracy based on historical variance
        """
        cv = model_info['std'] / model_info['mean'] if model_info['mean'] != 0 else 1
        accuracy = max(0.1, 1 - cv)  # Higher coefficient of variation = lower accuracy
        return accuracy

class AlertSystem:
    """
    Intelligent alert system for environmental anomalies
    """
    
    def __init__(self):
        self.alert_thresholds = {
            'ph': {'min': 6.5, 'max': 8.5, 'critical_min': 5.0, 'critical_max': 10.0},
            'dissolved_oxygen': {'min': 5.0, 'critical_min': 2.0},
            'temperature': {'min': 10, 'max': 25, 'critical_min': 5, 'critical_max': 35},
            'turbidity': {'max': 10, 'critical_max': 50},
            'conductivity': {'max': 1000, 'critical_max': 2000}
        }
        self.alert_history = []
        
    def check_environmental_parameters(self, current_data):
        """
        Check current environmental data against alert thresholds
        """
        alerts = []
        
        for parameter, value in current_data.items():
            if parameter in self.alert_thresholds:
                thresholds = self.alert_thresholds[parameter]
                alert = self._evaluate_parameter(parameter, value, thresholds)
                if alert:
                    alerts.append(alert)
        
        # Store alerts in history
        if alerts:
            self.alert_history.append({
                'timestamp': datetime.now(),
                'alerts': alerts,
                'data': current_data
            })
        
        return alerts
    
    def _evaluate_parameter(self, parameter, value, thresholds):
        """
        Evaluate single parameter against thresholds
        """
        alert = None
        
        # Check critical levels first
        if 'critical_min' in thresholds and value < thresholds['critical_min']:
            alert = {
                'parameter': parameter,
                'value': value,
                'level': 'CRITICAL',
                'message': f'{parameter} critically low: {value} < {thresholds["critical_min"]}',
                'threshold_violated': 'critical_min',
                'severity_score': 1.0
            }
        elif 'critical_max' in thresholds and value > thresholds['critical_max']:
            alert = {
                'parameter': parameter,
                'value': value,
                'level': 'CRITICAL',
                'message': f'{parameter} critically high: {value} > {thresholds["critical_max"]}',
                'threshold_violated': 'critical_max',
                'severity_score': 1.0
            }
        # Check warning levels
        elif 'min' in thresholds and value < thresholds['min']:
            alert = {
                'parameter': parameter,
                'value': value,
                'level': 'WARNING',
                'message': f'{parameter} below normal: {value} < {thresholds["min"]}',
                'threshold_violated': 'min',
                'severity_score': 0.6
            }
        elif 'max' in thresholds and value > thresholds['max']:
            alert = {
                'parameter': parameter,
                'value': value,
                'level': 'WARNING',
                'message': f'{parameter} above normal: {value} > {thresholds["max"]}',
                'threshold_violated': 'max',
                'severity_score': 0.6
            }
        
        return alert
    
    def get_alert_summary(self, hours_back=24):
        """
        Get summary of alerts from specified time period
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_alerts = [
            alert for alert in self.alert_history 
            if alert['timestamp'] > cutoff_time
        ]
        
        if not recent_alerts:
            return {'status': 'normal', 'alert_count': 0, 'alerts': []}
        
        critical_count = sum(
            len([a for a in alert['alerts'] if a['level'] == 'CRITICAL'])
            for alert in recent_alerts
        )
        warning_count = sum(
            len([a for a in alert['alerts'] if a['level'] == 'WARNING'])
            for alert in recent_alerts
        )
        
        overall_status = 'critical' if critical_count > 0 else 'warning' if warning_count > 0 else 'normal'
        
        return {
            'status': overall_status,
            'alert_count': len(recent_alerts),
            'critical_alerts': critical_count,
            'warning_alerts': warning_count,
            'recent_alerts': recent_alerts[-10:],  # Last 10 alerts
            'most_frequent_parameter': self._get_most_frequent_alert_parameter(recent_alerts)
        }
    
    def _get_most_frequent_alert_parameter(self, recent_alerts):
        """
        Find which parameter triggers alerts most frequently
        """
        parameter_counts = {}
        
        for alert_group in recent_alerts:
            for alert in alert_group['alerts']:
                param = alert['parameter']
                parameter_counts[param] = parameter_counts.get(param, 0) + 1
        
        if parameter_counts:
            return max(parameter_counts, key=parameter_counts.get)
        return None

# Utility functions for the machine learning module
def load_sample_data():
    """
    Generate sample environmental data for testing
    """
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    
    data = pd.DataFrame({
        'timestamp': dates,
        'ph': np.random.normal(7.2, 0.5, len(dates)),
        'dissolved_oxygen': np.random.normal(8.0, 1.2, len(dates)),
        'temperature': 20 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 2, len(dates)),
        'turbidity': np.random.exponential(5, len(dates)),
        'conductivity': np.random.normal(500, 100, len(dates)),
        'water_level': np.random.normal(150, 20, len(dates))
    })
    
    return data

def run_comprehensive_analysis():
    """
    Run a comprehensive analysis using all ML components
    """
    # Load sample data
    data = load_sample_data()
    
    # Initialize components
    quality_predictor = WaterQualityPredictor()
    pollution_detector = PollutionDetector()
    forecaster = EnvironmentalForecaster()
    alert_system = AlertSystem()
    
    print("🌊 Starting Comprehensive River Intelligence Analysis...")
    
    # Train water quality predictor
    features = data[['ph', 'dissolved_oxygen', 'temperature', 'turbidity', 'conductivity']]
    target = data['water_level']
    
    best_model, score = quality_predictor.train_models(features, target)
    print(f"✅ Water Quality Predictor trained. Best model: {best_model} (Score: {score:.4f})")
    
    # Train forecasting models
    for param in ['ph', 'dissolved_oxygen', 'temperature']:
        forecaster.train_forecasting_model(data, param)
    print("✅ Forecasting models trained for pH, DO, and Temperature")
    
    # Generate forecast
    ph_forecast = forecaster.forecast_parameter('ph', steps_ahead=7)
    print(f"📈 7-day pH forecast: {ph_forecast['predictions'][:3]}... (showing first 3 days)")
    
    # Check current conditions
    current_data = {
        'ph': 6.2,  # Slightly acidic
        'dissolved_oxygen': 4.5,  # Low oxygen
        'temperature': 28,
        'turbidity': 15,  # High turbidity
        'conductivity': 800
    }
    
    alerts = alert_system.check_environmental_parameters(current_data)
    if alerts:
        print(f"⚠️  {len(alerts)} alerts generated for current conditions")
        for alert in alerts:
            print(f"   - {alert['level']}: {alert['message']}")
    else:
        print("✅ No alerts - conditions within normal range")
    
    print("\n🎯 Analysis complete! All ML components operational.")
    
    return {
        'quality_predictor': quality_predictor,
        'pollution_detector': pollution_detector,
        'forecaster': forecaster,
        'alert_system': alert_system,
        'sample_data': data
    }

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run comprehensive analysis
    results = run_comprehensive_analysis()
