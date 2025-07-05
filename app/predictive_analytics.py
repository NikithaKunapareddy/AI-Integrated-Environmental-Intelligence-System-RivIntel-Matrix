"""
AI-Powered Predictive Analytics and Modeling Engine
Advanced predictive models for environmental forecasting
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso, BayesianRidge
from sklearn.svm import SVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import optuna
from scipy import signal
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class EnsemblePredictiveModel:
    """
    Advanced ensemble model for environmental parameter prediction
    """
    
    def __init__(self, prediction_horizon=24):
        self.prediction_horizon = prediction_horizon
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
            'extra_trees': ExtraTreesRegressor(n_estimators=200, max_depth=15, random_state=42),
            'gradient_boost': AdaBoostRegressor(n_estimators=100, random_state=42),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            'ridge': Ridge(alpha=1.0, random_state=42),
            'bayesian_ridge': BayesianRidge(),
            'svr_rbf': SVR(kernel='rbf', C=100, gamma=0.1),
            'svr_linear': SVR(kernel='linear', C=1),
            'knn': KNeighborsRegressor(n_neighbors=10, weights='distance'),
            'mlp': MLPRegressor(hidden_layer_sizes=(100, 50, 25), max_iter=2000, random_state=42),
            'gaussian_process': GaussianProcessRegressor(random_state=42),
            'decision_tree': DecisionTreeRegressor(max_depth=10, random_state=42)
        }
        
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        self.trained_models = {}
        self.feature_selector = None
        self.best_scaler = None
        self.model_weights = {}
        self.feature_importance = {}
        
    def create_time_features(self, timestamps):
        """Create time-based features from timestamps"""
        df = pd.DataFrame({'timestamp': pd.to_datetime(timestamps)})
        
        features = pd.DataFrame({
            'hour': df['timestamp'].dt.hour,
            'day_of_week': df['timestamp'].dt.dayofweek,
            'day_of_month': df['timestamp'].dt.day,
            'month': df['timestamp'].dt.month,
            'quarter': df['timestamp'].dt.quarter,
            'is_weekend': (df['timestamp'].dt.dayofweek >= 5).astype(int),
            'hour_sin': np.sin(2 * np.pi * df['timestamp'].dt.hour / 24),
            'hour_cos': np.cos(2 * np.pi * df['timestamp'].dt.hour / 24),
            'day_sin': np.sin(2 * np.pi * df['timestamp'].dt.dayofyear / 365),
            'day_cos': np.cos(2 * np.pi * df['timestamp'].dt.dayofyear / 365),
            'month_sin': np.sin(2 * np.pi * df['timestamp'].dt.month / 12),
            'month_cos': np.cos(2 * np.pi * df['timestamp'].dt.month / 12)
        })
        
        return features
    
    def create_lag_features(self, series, lags=[1, 2, 3, 6, 12, 24]):
        """Create lagged features from time series"""
        features = pd.DataFrame()
        
        for lag in lags:
            features[f'lag_{lag}'] = series.shift(lag)
            features[f'lag_{lag}_diff'] = series.diff(lag)
            
        # Rolling statistics
        for window in [3, 6, 12, 24]:
            if window <= len(series):
                features[f'rolling_mean_{window}'] = series.rolling(window=window).mean()
                features[f'rolling_std_{window}'] = series.rolling(window=window).std()
                features[f'rolling_min_{window}'] = series.rolling(window=window).min()
                features[f'rolling_max_{window}'] = series.rolling(window=window).max()
                features[f'rolling_median_{window}'] = series.rolling(window=window).median()
        
        # Exponential moving averages
        for alpha in [0.1, 0.3, 0.5]:
            features[f'ema_{alpha}'] = series.ewm(alpha=alpha).mean()
        
        return features
    
    def create_spectral_features(self, series, n_features=5):
        """Create frequency domain features using FFT"""
        if len(series) < 20:
            return pd.DataFrame()
        
        # Apply FFT
        fft_values = np.fft.fft(series.fillna(series.mean()))
        fft_freq = np.fft.fftfreq(len(series))
        
        # Get dominant frequencies
        power_spectrum = np.abs(fft_values) ** 2
        top_freq_indices = np.argsort(power_spectrum)[-n_features:]
        
        features = pd.DataFrame()
        for i, idx in enumerate(top_freq_indices):
            features[f'fft_freq_{i}'] = [fft_freq[idx]] * len(series)
            features[f'fft_power_{i}'] = [power_spectrum[idx]] * len(series)
            features[f'fft_real_{i}'] = [fft_values[idx].real] * len(series)
            features[f'fft_imag_{i}'] = [fft_values[idx].imag] * len(series)
        
        return features
    
    def create_interaction_features(self, df, max_interactions=10):
        """Create interaction features between variables"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        interaction_features = pd.DataFrame(index=df.index)
        
        interaction_count = 0
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                if interaction_count >= max_interactions:
                    break
                    
                # Multiplicative interaction
                interaction_features[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                
                # Ratio interaction (with small epsilon to avoid division by zero)
                interaction_features[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
                
                # Difference interaction
                interaction_features[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
                
                interaction_count += 1
        
        return interaction_features
    
    def prepare_features(self, data, target_column, timestamps=None):
        """Comprehensive feature engineering pipeline"""
        # Base features
        features = data.drop(columns=[target_column]).copy()
        target = data[target_column].copy()
        
        # Time-based features
        if timestamps is not None:
            time_features = self.create_time_features(timestamps)
            features = pd.concat([features, time_features], axis=1)
        
        # Lag features for target variable
        lag_features = self.create_lag_features(target)
        features = pd.concat([features, lag_features], axis=1)
        
        # Lag features for other variables
        for col in data.columns:
            if col != target_column:
                col_lag_features = self.create_lag_features(data[col], lags=[1, 3, 6])
                col_lag_features.columns = [f'{col}_{c}' for c in col_lag_features.columns]
                features = pd.concat([features, col_lag_features], axis=1)
        
        # Spectral features
        spectral_features = self.create_spectral_features(target)
        features = pd.concat([features, spectral_features], axis=1)
        
        # Interaction features
        interaction_features = self.create_interaction_features(features[data.columns[:-1]])
        features = pd.concat([features, interaction_features], axis=1)
        
        # Statistical features
        for col in features.select_dtypes(include=[np.number]).columns:
            if col not in ['hour', 'day_of_week', 'day_of_month', 'month', 'quarter', 'is_weekend']:
                features[f'{col}_squared'] = features[col] ** 2
                features[f'{col}_sqrt'] = np.sqrt(np.abs(features[col]))
                features[f'{col}_log'] = np.log1p(np.abs(features[col]))
        
        # Remove rows with too many NaN values
        features = features.dropna(thresh=len(features.columns) * 0.7)
        target = target.loc[features.index]
        
        # Fill remaining NaN values
        features = features.fillna(features.median())
        
        return features, target
    
    def optimize_hyperparameters(self, X, y, model_name, n_trials=50):
        """Optimize hyperparameters using Optuna"""
        def objective(trial):
            if model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
                }
                model = RandomForestRegressor(**params, random_state=42)
                
            elif model_name == 'svr_rbf':
                params = {
                    'C': trial.suggest_loguniform('C', 0.1, 1000),
                    'gamma': trial.suggest_loguniform('gamma', 1e-6, 1e-1),
                    'epsilon': trial.suggest_uniform('epsilon', 0.01, 1.0)
                }
                model = SVR(kernel='rbf', **params)
                
            elif model_name == 'mlp':
                params = {
                    'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', 
                                                                   [(50,), (100,), (50, 25), (100, 50), (100, 50, 25)]),
                    'alpha': trial.suggest_loguniform('alpha', 1e-6, 1e-1),
                    'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 1e-5, 1e-1)
                }
                model = MLPRegressor(**params, max_iter=1000, random_state=42)
                
            else:
                return -np.inf  # Skip optimization for other models
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
            return scores.mean()
        
        try:
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            return study.best_params
        except Exception as e:
            print(f"Hyperparameter optimization failed for {model_name}: {str(e)}")
            return {}
    
    def train_ensemble(self, data, target_column, timestamps=None, optimize_hyperparams=True):
        """Train ensemble of models with comprehensive feature engineering"""
        print("🔧 Preparing features for ensemble training...")
        
        # Feature engineering
        X, y = self.prepare_features(data, target_column, timestamps)
        
        print(f"📊 Feature engineering complete: {X.shape[1]} features created")
        
        # Feature selection
        if X.shape[1] > 50:  # Only if we have many features
            self.feature_selector = SelectKBest(score_func=mutual_info_regression, k=min(50, X.shape[1]))
            X = self.feature_selector.fit_transform(X, y)
            print(f"🎯 Feature selection: reduced to {X.shape[1]} features")
        
        # Find best scaler
        best_score = -np.inf
        for scaler_name, scaler in self.scalers.items():
            X_scaled = scaler.fit_transform(X)
            
            # Quick test with a simple model
            test_model = Ridge(random_state=42)
            tscv = TimeSeriesSplit(n_splits=3)
            scores = cross_val_score(test_model, X_scaled, y, cv=tscv, scoring='neg_mean_squared_error')
            avg_score = scores.mean()
            
            if avg_score > best_score:
                best_score = avg_score
                self.best_scaler = scaler_name
        
        print(f"⚖️  Best scaler: {self.best_scaler}")
        
        # Scale features
        X_scaled = self.scalers[self.best_scaler].fit_transform(X)
        
        # Train models
        model_scores = {}
        
        for model_name, model in self.models.items():
            try:
                print(f"🤖 Training {model_name}...")
                
                # Hyperparameter optimization for selected models
                if optimize_hyperparams and model_name in ['random_forest', 'svr_rbf', 'mlp']:
                    best_params = self.optimize_hyperparameters(X_scaled, y, model_name, n_trials=20)
                    if best_params:
                        model.set_params(**best_params)
                        print(f"   ✅ Optimized hyperparameters: {best_params}")
                
                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=5)
                scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='neg_mean_squared_error')
                avg_score = scores.mean()
                std_score = scores.std()
                
                model_scores[model_name] = {
                    'score': avg_score,
                    'std': std_score,
                    'stability': 1 / (1 + std_score)  # Higher stability = lower std
                }
                
                # Train on full dataset
                model.fit(X_scaled, y)
                self.trained_models[model_name] = model
                
                print(f"   📈 Score: {avg_score:.4f} ± {std_score:.4f}")
                
            except Exception as e:
                print(f"   ❌ Failed to train {model_name}: {str(e)}")
        
        # Calculate ensemble weights based on performance and stability
        total_weight = 0
        for model_name, score_info in model_scores.items():
            # Weight = performance * stability
            weight = np.exp(score_info['score']) * score_info['stability']
            self.model_weights[model_name] = weight
            total_weight += weight
        
        # Normalize weights
        for model_name in self.model_weights:
            self.model_weights[model_name] /= total_weight
        
        print(f"🎭 Ensemble weights calculated:")
        for model_name, weight in self.model_weights.items():
            print(f"   {model_name}: {weight:.3f}")
        
        # Calculate feature importance for tree-based models
        self._calculate_feature_importance(X.shape[1] if hasattr(X, 'shape') else len(X.columns))
        
        return model_scores
    
    def _calculate_feature_importance(self, n_features):
        """Calculate feature importance from tree-based models"""
        tree_models = ['random_forest', 'extra_trees', 'decision_tree']
        importance_sum = np.zeros(n_features)
        model_count = 0
        
        for model_name in tree_models:
            if model_name in self.trained_models:
                try:
                    importance = self.trained_models[model_name].feature_importances_
                    importance_sum += importance
                    model_count += 1
                except AttributeError:
                    continue
        
        if model_count > 0:
            self.feature_importance = importance_sum / model_count
        
    def predict(self, new_data, timestamps=None, return_uncertainty=True):
        """Make predictions using the trained ensemble"""
        if not self.trained_models:
            raise ValueError("No trained models available. Please train the ensemble first.")
        
        # Prepare features (this is tricky without the target column)
        # For now, assume new_data has the same structure as training data minus target
        X = new_data.copy()
        
        # Apply same preprocessing as training
        if timestamps is not None:
            time_features = self.create_time_features(timestamps)
            X = pd.concat([X, time_features], axis=1)
        
        # Fill missing values
        X = X.fillna(X.median())
        
        # Feature selection
        if self.feature_selector:
            X_selected = self.feature_selector.transform(X)
        else:
            X_selected = X.values
        
        # Scale features
        X_scaled = self.scalers[self.best_scaler].transform(X_selected)
        
        # Get predictions from all models
        predictions = {}
        weighted_prediction = np.zeros(len(X_scaled))
        
        for model_name, model in self.trained_models.items():
            try:
                pred = model.predict(X_scaled)
                predictions[model_name] = pred
                weight = self.model_weights.get(model_name, 0)
                weighted_prediction += weight * pred
            except Exception as e:
                print(f"❌ Prediction failed for {model_name}: {str(e)}")
        
        result = {
            'ensemble_prediction': weighted_prediction,
            'individual_predictions': predictions
        }
        
        if return_uncertainty:
            # Calculate prediction uncertainty
            pred_values = list(predictions.values())
            if len(pred_values) > 1:
                prediction_std = np.std(pred_values, axis=0)
                result['uncertainty'] = prediction_std
                result['confidence_interval'] = {
                    'lower': weighted_prediction - 1.96 * prediction_std,
                    'upper': weighted_prediction + 1.96 * prediction_std
                }
        
        return result
    
    def forecast_horizon(self, data, target_column, timestamps=None, steps_ahead=24):
        """Forecast multiple steps ahead"""
        if not self.trained_models:
            raise ValueError("Ensemble must be trained before forecasting")
        
        forecasts = []
        current_data = data.copy()
        current_timestamps = list(timestamps) if timestamps is not None else None
        
        for step in range(steps_ahead):
            # Prepare data for prediction (excluding target column for prediction)
            pred_data = current_data.drop(columns=[target_column])
            
            # Make prediction
            if current_timestamps:
                pred_result = self.predict(pred_data.tail(1), [current_timestamps[-1]], return_uncertainty=True)
            else:
                pred_result = self.predict(pred_data.tail(1), return_uncertainty=True)
            
            forecast_value = pred_result['ensemble_prediction'][0]
            uncertainty = pred_result.get('uncertainty', [0])[0]
            
            forecasts.append({
                'step': step + 1,
                'prediction': forecast_value,
                'uncertainty': uncertainty,
                'lower_bound': forecast_value - 1.96 * uncertainty,
                'upper_bound': forecast_value + 1.96 * uncertainty
            })
            
            # Update data for next prediction
            new_row = current_data.iloc[-1].copy()
            new_row[target_column] = forecast_value
            
            current_data = pd.concat([current_data, new_row.to_frame().T], ignore_index=True)
            
            # Update timestamps
            if current_timestamps:
                from datetime import timedelta
                next_time = pd.to_datetime(current_timestamps[-1]) + timedelta(hours=1)
                current_timestamps.append(next_time)
        
        return forecasts

def generate_comprehensive_test_data():
    """Generate comprehensive test data for the predictive model"""
    np.random.seed(42)
    
    # Generate 2 years of hourly data
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='H')
    n_points = len(dates)
    
    # Create realistic environmental data with patterns
    data = pd.DataFrame({
        'timestamp': dates,
        'ph': 7.0 + 0.3 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 7)) + np.random.normal(0, 0.2, n_points),
        'dissolved_oxygen': 8.0 + 2 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 365)) + np.random.normal(0, 0.5, n_points),
        'conductivity': 500 + 100 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 30)) + np.random.normal(0, 20, n_points),
        'turbidity': 5 + 3 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 7)) + np.random.exponential(2, n_points),
    })
    
    # Create target variable (water temperature) with realistic patterns
    data['temperature'] = (
        15 +  # Base temperature
        10 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 365)) +  # Seasonal variation
        5 * np.sin(2 * np.pi * np.arange(n_points) / 24) +  # Daily variation
        0.1 * data['ph'] +  # Slight pH influence
        0.05 * data['dissolved_oxygen'] +  # Slight DO influence
        np.random.normal(0, 1, n_points)  # Random noise
    )
    
    return data

async def run_predictive_modeling_demo():
    """Comprehensive demonstration of the predictive modeling system"""
    print("🚀 Starting Advanced Predictive Modeling Demo...")
    
    # Generate test data
    print("📊 Generating comprehensive test data...")
    data = generate_comprehensive_test_data()
    print(f"✅ Generated {len(data)} data points with {len(data.columns)-1} features")
    
    # Initialize ensemble model
    ensemble = EnsemblePredictiveModel(prediction_horizon=24)
    
    # Split data for training and testing
    split_point = int(len(data) * 0.8)
    train_data = data[:split_point].copy()
    test_data = data[split_point:].copy()
    
    print(f"📊 Training set: {len(train_data)} points")
    print(f"📊 Test set: {len(test_data)} points")
    
    # Train ensemble
    print("\n🎓 Training ensemble model...")
    model_scores = ensemble.train_ensemble(
        train_data.drop(columns=['timestamp']), 
        target_column='temperature',
        timestamps=train_data['timestamp'],
        optimize_hyperparams=True
    )
    
    print(f"✅ Ensemble training complete. {len(ensemble.trained_models)} models trained.")
    
    # Make predictions on test set
    print("\n🔮 Making predictions on test set...")
    test_features = test_data.drop(columns=['temperature', 'timestamp'])
    predictions = ensemble.predict(test_features, test_data['timestamp'], return_uncertainty=True)
    
    # Calculate performance metrics
    y_true = test_data['temperature'].values
    y_pred = predictions['ensemble_prediction']
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"📈 Prediction Performance:")
    print(f"   - Mean Absolute Error: {mae:.4f}")
    print(f"   - Root Mean Square Error: {rmse:.4f}")
    print(f"   - R² Score: {r2:.4f}")
    
    # Generate forecast
    print("\n🔮 Generating 24-hour forecast...")
    forecast = ensemble.forecast_horizon(
        train_data.drop(columns=['timestamp']).tail(100),  # Use last 100 points
        target_column='temperature',
        timestamps=train_data['timestamp'].tail(100),
        steps_ahead=24
    )
    
    print(f"📅 24-hour forecast generated:")
    for i in range(0, min(6, len(forecast))):  # Show first 6 hours
        f = forecast[i]
        print(f"   Hour {f['step']}: {f['prediction']:.2f}°C ± {f['uncertainty']:.2f}")
    
    # Model analysis
    print(f"\n🎭 Ensemble Model Analysis:")
    print(f"   - Best performing models:")
    sorted_weights = sorted(ensemble.model_weights.items(), key=lambda x: x[1], reverse=True)
    for model_name, weight in sorted_weights[:3]:
        print(f"     {model_name}: {weight:.3f}")
    
    if hasattr(ensemble, 'feature_importance') and len(ensemble.feature_importance) > 0:
        print(f"   - Feature importance calculated for {len(ensemble.feature_importance)} features")
    
    print("\n🎯 Predictive modeling demo completed successfully!")
    
    return {
        'ensemble': ensemble,
        'performance': {'mae': mae, 'rmse': rmse, 'r2': r2},
        'forecast': forecast,
        'test_data': test_data,
        'predictions': predictions
    }

if __name__ == "__main__":
    import asyncio
    
    # Run the comprehensive demo
    results = asyncio.run(run_predictive_modeling_demo())
