"""
Advanced Data Processing and Analytics Engine
Real-time data processing pipeline for environmental monitoring
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter, find_peaks
import asyncio
import aiofiles
import json
from datetime import datetime, timedelta
import sqlite3
import threading
import queue
import time
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

class RealTimeDataProcessor:
    """
    High-performance real-time data processing engine
    """
    
    def __init__(self, buffer_size=1000, processing_interval=1.0):
        self.buffer_size = buffer_size
        self.processing_interval = processing_interval
        self.data_buffer = deque(maxlen=buffer_size)
        self.processed_buffer = deque(maxlen=buffer_size)
        self.processing_queue = queue.Queue()
        self.is_running = False
        self.processor_thread = None
        self.callbacks = defaultdict(list)
        self.statistics = {
            'processed_records': 0,
            'anomalies_detected': 0,
            'processing_errors': 0,
            'average_processing_time': 0.0
        }
        
    def start_processing(self):
        """Start the real-time processing engine"""
        if not self.is_running:
            self.is_running = True
            self.processor_thread = threading.Thread(target=self._processing_loop)
            self.processor_thread.daemon = True
            self.processor_thread.start()
            print("🚀 Real-time data processor started")
    
    def stop_processing(self):
        """Stop the processing engine"""
        self.is_running = False
        if self.processor_thread:
            self.processor_thread.join()
        print("⏹️  Real-time data processor stopped")
    
    def add_data_point(self, data_point):
        """Add new data point to processing queue"""
        if isinstance(data_point, dict):
            data_point['timestamp'] = data_point.get('timestamp', datetime.now())
            self.processing_queue.put(data_point)
        else:
            raise ValueError("Data point must be a dictionary")
    
    def register_callback(self, event_type, callback_function):
        """Register callback for specific events"""
        self.callbacks[event_type].append(callback_function)
    
    def _processing_loop(self):
        """Main processing loop"""
        while self.is_running:
            try:
                # Process queued data points
                batch_data = []
                while not self.processing_queue.empty() and len(batch_data) < 10:
                    batch_data.append(self.processing_queue.get_nowait())
                
                if batch_data:
                    start_time = time.time()
                    self._process_batch(batch_data)
                    processing_time = time.time() - start_time
                    
                    # Update statistics
                    self.statistics['processed_records'] += len(batch_data)
                    self.statistics['average_processing_time'] = (
                        self.statistics['average_processing_time'] * 0.9 + processing_time * 0.1
                    )
                
                time.sleep(self.processing_interval)
                
            except Exception as e:
                self.statistics['processing_errors'] += 1
                print(f"❌ Processing error: {str(e)}")
                time.sleep(1.0)  # Brief pause on error
    
    def _process_batch(self, batch_data):
        """Process a batch of data points"""
        for data_point in batch_data:
            # Add to data buffer
            self.data_buffer.append(data_point)
            
            # Perform real-time analysis
            processed_point = self._analyze_data_point(data_point)
            self.processed_buffer.append(processed_point)
            
            # Check for anomalies
            if self._detect_anomaly(processed_point):
                self.statistics['anomalies_detected'] += 1
                self._trigger_callbacks('anomaly_detected', processed_point)
            
            # Trigger data processed callbacks
            self._trigger_callbacks('data_processed', processed_point)
    
    def _analyze_data_point(self, data_point):
        """Analyze individual data point"""
        analyzed = data_point.copy()
        
        # Calculate rolling statistics if enough data points
        if len(self.data_buffer) >= 10:
            recent_data = list(self.data_buffer)[-10:]
            
            for key, value in data_point.items():
                if isinstance(value, (int, float)) and key != 'timestamp':
                    recent_values = [d.get(key, 0) for d in recent_data if isinstance(d.get(key), (int, float))]
                    
                    if recent_values:
                        analyzed[f'{key}_mean'] = np.mean(recent_values)
                        analyzed[f'{key}_std'] = np.std(recent_values)
                        analyzed[f'{key}_trend'] = self._calculate_trend(recent_values)
                        analyzed[f'{key}_z_score'] = (value - np.mean(recent_values)) / (np.std(recent_values) + 1e-8)
        
        # Add quality score
        analyzed['quality_score'] = self._calculate_quality_score(analyzed)
        
        return analyzed
    
    def _calculate_trend(self, values):
        """Calculate trend direction for recent values"""
        if len(values) < 3:
            return 0
        
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        return slope
    
    def _calculate_quality_score(self, data_point):
        """Calculate overall data quality score"""
        score = 1.0
        
        # Check for missing values
        missing_ratio = sum(1 for v in data_point.values() if v is None or (isinstance(v, float) and np.isnan(v)))
        missing_ratio /= len(data_point)
        score -= missing_ratio * 0.3
        
        # Check for extreme Z-scores
        z_scores = [v for k, v in data_point.items() if k.endswith('_z_score')]
        if z_scores:
            extreme_z_count = sum(1 for z in z_scores if abs(z) > 3)
            score -= (extreme_z_count / len(z_scores)) * 0.2
        
        return max(0.0, min(1.0, score))
    
    def _detect_anomaly(self, processed_point):
        """Detect anomalies in processed data"""
        # Simple anomaly detection based on Z-scores
        z_scores = [v for k, v in processed_point.items() if k.endswith('_z_score')]
        
        if z_scores:
            max_z_score = max(abs(z) for z in z_scores)
            return max_z_score > 2.5  # Anomaly threshold
        
        return False
    
    def _trigger_callbacks(self, event_type, data):
        """Trigger registered callbacks for event type"""
        for callback in self.callbacks[event_type]:
            try:
                callback(data)
            except Exception as e:
                print(f"❌ Callback error for {event_type}: {str(e)}")
    
    def get_statistics(self):
        """Get current processing statistics"""
        return self.statistics.copy()
    
    def get_recent_data(self, count=10):
        """Get recent processed data points"""
        return list(self.processed_buffer)[-count:]

class AdvancedAnalytics:
    """
    Advanced analytics and pattern recognition
    """
    
    def __init__(self):
        self.patterns = {}
        self.seasonal_models = {}
        self.correlation_matrix = None
        
    def detect_patterns(self, data, parameter, window_size=24):
        """
        Detect recurring patterns in time series data
        """
        if len(data) < window_size * 2:
            return {"error": "Insufficient data for pattern detection"}
        
        values = [d.get(parameter, 0) for d in data if isinstance(d.get(parameter), (int, float))]
        
        if len(values) < window_size:
            return {"error": f"Insufficient {parameter} data"}
        
        # Smooth the data
        smoothed = savgol_filter(values, min(11, len(values)//4*2+1), 3)
        
        # Find peaks and valleys
        peaks, _ = find_peaks(smoothed, distance=window_size//4)
        valleys, _ = find_peaks(-smoothed, distance=window_size//4)
        
        # Calculate pattern metrics
        patterns = {
            'peaks': {
                'count': len(peaks),
                'positions': peaks.tolist(),
                'values': [smoothed[i] for i in peaks],
                'average_interval': np.mean(np.diff(peaks)) if len(peaks) > 1 else 0
            },
            'valleys': {
                'count': len(valleys),
                'positions': valleys.tolist(),
                'values': [smoothed[i] for i in valleys],
                'average_interval': np.mean(np.diff(valleys)) if len(valleys) > 1 else 0
            },
            'trend': {
                'overall_slope': self._calculate_overall_trend(values),
                'volatility': np.std(np.diff(values)),
                'mean_value': np.mean(values),
                'range': np.max(values) - np.min(values)
            },
            'cyclical': self._detect_cyclical_patterns(smoothed),
            'seasonality': self._detect_seasonality(values, window_size)
        }
        
        self.patterns[parameter] = patterns
        return patterns
    
    def _calculate_overall_trend(self, values):
        """Calculate overall trend slope"""
        x = np.arange(len(values))
        slope, _, r_value, _, _ = stats.linregress(x, values)
        return {'slope': slope, 'r_squared': r_value**2}
    
    def _detect_cyclical_patterns(self, smoothed_values):
        """Detect cyclical patterns using autocorrelation"""
        if len(smoothed_values) < 20:
            return {'detected': False}
        
        # Calculate autocorrelation
        autocorr = np.correlate(smoothed_values, smoothed_values, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find significant peaks in autocorrelation
        peaks, properties = find_peaks(autocorr[1:], height=0.3, distance=5)
        
        if len(peaks) > 0:
            dominant_cycle = peaks[0] + 1  # +1 because we started from index 1
            return {
                'detected': True,
                'cycle_length': dominant_cycle,
                'strength': autocorr[dominant_cycle],
                'all_cycles': (peaks + 1).tolist()
            }
        
        return {'detected': False}
    
    def _detect_seasonality(self, values, period=24):
        """Detect seasonal patterns"""
        if len(values) < period * 2:
            return {'detected': False}
        
        # Reshape data into periods
        full_periods = len(values) // period
        reshaped = np.array(values[:full_periods * period]).reshape(full_periods, period)
        
        # Calculate seasonal profile
        seasonal_profile = np.mean(reshaped, axis=0)
        seasonal_std = np.std(reshaped, axis=0)
        
        # Calculate seasonality strength
        seasonal_variance = np.var(seasonal_profile)
        total_variance = np.var(values)
        seasonality_strength = seasonal_variance / (total_variance + 1e-8)
        
        return {
            'detected': seasonality_strength > 0.1,
            'strength': seasonality_strength,
            'profile': seasonal_profile.tolist(),
            'profile_std': seasonal_std.tolist(),
            'period': period
        }
    
    def correlation_analysis(self, data, parameters):
        """
        Perform correlation analysis between parameters
        """
        # Create DataFrame for correlation analysis
        df_data = []
        for point in data:
            row = {}
            for param in parameters:
                if param in point and isinstance(point[param], (int, float)):
                    row[param] = point[param]
            if len(row) == len(parameters):  # Only include complete records
                df_data.append(row)
        
        if len(df_data) < 10:
            return {"error": "Insufficient complete data for correlation analysis"}
        
        df = pd.DataFrame(df_data)
        
        # Calculate correlation matrix
        correlation_matrix = df.corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(parameters)):
            for j in range(i+1, len(parameters)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Strong correlation threshold
                    strong_correlations.append({
                        'parameter1': parameters[i],
                        'parameter2': parameters[j],
                        'correlation': corr_value,
                        'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate'
                    })
        
        self.correlation_matrix = correlation_matrix
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': strong_correlations,
            'summary': {
                'total_correlations_found': len(strong_correlations),
                'strongest_correlation': max(strong_correlations, key=lambda x: abs(x['correlation'])) if strong_correlations else None
            }
        }
    
    def anomaly_scoring(self, data, parameter, method='isolation_forest'):
        """
        Advanced anomaly detection using multiple methods
        """
        values = [d.get(parameter, 0) for d in data if isinstance(d.get(parameter), (int, float))]
        
        if len(values) < 20:
            return {"error": "Insufficient data for anomaly detection"}
        
        values = np.array(values)
        
        # Statistical method (Z-score based)
        z_scores = np.abs(stats.zscore(values))
        statistical_anomalies = np.where(z_scores > 2.5)[0]
        
        # IQR method
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_anomalies = np.where((values < lower_bound) | (values > upper_bound))[0]
        
        # Moving average method
        window_size = min(10, len(values) // 5)
        moving_avg = pd.Series(values).rolling(window=window_size, center=True).mean()
        moving_std = pd.Series(values).rolling(window=window_size, center=True).std()
        
        deviation_scores = np.abs(values - moving_avg) / (moving_std + 1e-8)
        moving_avg_anomalies = np.where(deviation_scores > 2.0)[0]
        
        # Combine results
        all_anomalies = set(statistical_anomalies) | set(iqr_anomalies) | set(moving_avg_anomalies)
        
        anomaly_details = []
        for idx in sorted(all_anomalies):
            if not np.isnan(deviation_scores[idx]):
                anomaly_details.append({
                    'index': int(idx),
                    'value': float(values[idx]),
                    'z_score': float(z_scores[idx]),
                    'deviation_score': float(deviation_scores[idx]),
                    'methods_detected': [
                        method for method, anomalies in [
                            ('statistical', statistical_anomalies),
                            ('iqr', iqr_anomalies),
                            ('moving_average', moving_avg_anomalies)
                        ] if idx in anomalies
                    ]
                })
        
        return {
            'total_anomalies': len(anomaly_details),
            'anomaly_rate': len(anomaly_details) / len(values),
            'anomalies': anomaly_details[:20],  # Limit to first 20 for performance
            'statistics': {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'Q1': float(Q1),
                'Q3': float(Q3)
            }
        }

class DataPersistenceManager:
    """
    Advanced data persistence and retrieval system
    """
    
    def __init__(self, db_path='river_intelligence.db'):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database with optimized schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create main data table with indexes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS environmental_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                parameter_name TEXT NOT NULL,
                value REAL NOT NULL,
                quality_score REAL DEFAULT 1.0,
                is_anomaly BOOLEAN DEFAULT FALSE,
                source TEXT DEFAULT 'sensor',
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON environmental_data(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_parameter ON environmental_data(parameter_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_anomaly ON environmental_data(is_anomaly)')
        
        # Create analysis results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_type TEXT NOT NULL,
                parameter_name TEXT NOT NULL,
                result_data TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                expires_at DATETIME
            )
        ''')
        
        # Create alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                parameters_involved TEXT,
                is_resolved BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                resolved_at DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def store_data_batch(self, data_batch):
        """Store batch of data points asynchronously"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        records = []
        for data_point in data_batch:
            timestamp = data_point.get('timestamp', datetime.now())
            for key, value in data_point.items():
                if key != 'timestamp' and isinstance(value, (int, float)):
                    records.append((
                        timestamp,
                        key,
                        value,
                        data_point.get('quality_score', 1.0),
                        data_point.get('is_anomaly', False),
                        data_point.get('source', 'sensor'),
                        json.dumps(data_point.get('metadata', {}))
                    ))
        
        cursor.executemany('''
            INSERT INTO environmental_data 
            (timestamp, parameter_name, value, quality_score, is_anomaly, source, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', records)
        
        conn.commit()
        conn.close()
        
        return len(records)
    
    def query_data(self, parameters=None, start_date=None, end_date=None, limit=1000):
        """Query environmental data with filters"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT timestamp, parameter_name, value, quality_score, is_anomaly, source, metadata
            FROM environmental_data
            WHERE 1=1
        '''
        params = []
        
        if parameters:
            placeholders = ','.join(['?' for _ in parameters])
            query += f' AND parameter_name IN ({placeholders})'
            params.extend(parameters)
        
        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND timestamp <= ?'
            params.append(end_date)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_data_summary(self, hours_back=24):
        """Get summary statistics for recent data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # Get record counts by parameter
        cursor.execute('''
            SELECT parameter_name, COUNT(*) as count, AVG(value) as avg_value, 
                   MIN(value) as min_value, MAX(value) as max_value
            FROM environmental_data 
            WHERE timestamp > ? 
            GROUP BY parameter_name
        ''', (cutoff_time,))
        
        parameter_stats = cursor.fetchall()
        
        # Get anomaly count
        cursor.execute('''
            SELECT COUNT(*) FROM environmental_data 
            WHERE timestamp > ? AND is_anomaly = TRUE
        ''', (cutoff_time,))
        
        anomaly_count = cursor.fetchone()[0]
        
        # Get total records
        cursor.execute('''
            SELECT COUNT(*) FROM environmental_data 
            WHERE timestamp > ?
        ''', (cutoff_time,))
        
        total_records = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'time_period_hours': hours_back,
            'total_records': total_records,
            'anomaly_count': anomaly_count,
            'anomaly_rate': anomaly_count / total_records if total_records > 0 else 0,
            'parameter_statistics': [
                {
                    'parameter': row[0],
                    'record_count': row[1],
                    'average_value': row[2],
                    'min_value': row[3],
                    'max_value': row[4]
                }
                for row in parameter_stats
            ]
        }

# Example usage and testing functions
def generate_sample_stream():
    """Generate sample data stream for testing"""
    parameters = ['ph', 'dissolved_oxygen', 'temperature', 'turbidity', 'conductivity']
    
    while True:
        data_point = {
            'timestamp': datetime.now(),
            'ph': np.random.normal(7.2, 0.3),
            'dissolved_oxygen': np.random.normal(8.0, 1.0),
            'temperature': 20 + 5 * np.sin(time.time() / 3600) + np.random.normal(0, 1),
            'turbidity': np.random.exponential(3),
            'conductivity': np.random.normal(500, 50),
            'source': 'test_sensor'
        }
        
        # Occasionally introduce anomalies
        if np.random.random() < 0.05:  # 5% chance of anomaly
            anomaly_param = np.random.choice(parameters)
            if anomaly_param == 'ph':
                data_point[anomaly_param] = np.random.choice([5.0, 9.5])  # Extreme pH
            elif anomaly_param == 'dissolved_oxygen':
                data_point[anomaly_param] = np.random.uniform(1, 3)  # Low oxygen
            data_point['is_anomaly'] = True
        
        yield data_point

async def run_data_processing_demo():
    """Demonstrate the data processing system"""
    print("🔄 Starting Advanced Data Processing Demo...")
    
    # Initialize components
    processor = RealTimeDataProcessor(buffer_size=100, processing_interval=0.5)
    analytics = AdvancedAnalytics()
    persistence = DataPersistenceManager()
    
    # Register callbacks
    def on_anomaly_detected(data):
        print(f"🚨 Anomaly detected: {data.get('quality_score', 'N/A')} quality score")
    
    def on_data_processed(data):
        print(f"📊 Processed: {len([k for k in data.keys() if not k.endswith('_mean')])} parameters")
    
    processor.register_callback('anomaly_detected', on_anomaly_detected)
    processor.register_callback('data_processed', on_data_processed)
    
    # Start processing
    processor.start_processing()
    
    # Generate and process sample data
    sample_generator = generate_sample_stream()
    processed_data = []
    
    print("⏳ Processing sample data for 10 seconds...")
    end_time = time.time() + 10
    
    while time.time() < end_time:
        data_point = next(sample_generator)
        processor.add_data_point(data_point)
        processed_data.append(data_point)
        await asyncio.sleep(0.1)
    
    # Wait for processing to complete
    await asyncio.sleep(2)
    
    # Stop processor
    processor.stop_processing()
    
    # Run analytics
    print("\n📈 Running advanced analytics...")
    
    if len(processed_data) > 50:
        # Pattern detection
        patterns = analytics.detect_patterns(processed_data, 'temperature')
        print(f"🔍 Temperature patterns: {patterns.get('trend', {}).get('overall_slope', 'N/A')}")
        
        # Correlation analysis
        correlations = analytics.correlation_analysis(processed_data, ['ph', 'dissolved_oxygen', 'temperature'])
        strong_corr_count = len(correlations.get('strong_correlations', []))
        print(f"🔗 Found {strong_corr_count} strong parameter correlations")
        
        # Anomaly scoring
        anomalies = analytics.anomaly_scoring(processed_data, 'ph')
        print(f"⚠️  Detected {anomalies.get('total_anomalies', 0)} pH anomalies")
    
    # Store data
    print("\n💾 Storing processed data...")
    stored_count = await persistence.store_data_batch(processed_data[-20:])  # Store last 20 records
    print(f"✅ Stored {stored_count} data points to database")
    
    # Get summary
    summary = persistence.get_data_summary(hours_back=1)
    print(f"📋 Database summary: {summary['total_records']} total records, {summary['anomaly_count']} anomalies")
    
    # Display statistics
    stats = processor.get_statistics()
    print(f"\n📊 Processing Statistics:")
    print(f"   - Records processed: {stats['processed_records']}")
    print(f"   - Anomalies detected: {stats['anomalies_detected']}")
    print(f"   - Average processing time: {stats['average_processing_time']:.4f}s")
    
    print("\n🎯 Advanced data processing demo completed!")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_data_processing_demo())
