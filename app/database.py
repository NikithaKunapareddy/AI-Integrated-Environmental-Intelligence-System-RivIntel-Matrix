"""
Database models and operations for RiverMind system
"""
import sqlite3
import json
from datetime import datetime
from contextlib import contextmanager
import os

DATABASE_PATH = 'rivermind.db'

class DatabaseManager:
    """Manages database operations for the RiverMind system"""
    
    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = db_path
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize database with required tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Environmental data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS environmental_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    temperature REAL NOT NULL,
                    ph REAL NOT NULL,
                    flow REAL NOT NULL,
                    emotion VARCHAR(20),
                    location VARCHAR(100),
                    source VARCHAR(50) DEFAULT 'manual'
                )
            ''')
            
            # User profiles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id VARCHAR(50) PRIMARY KEY,
                    preferences TEXT,  -- JSON string
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_active DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type VARCHAR(20) NOT NULL,
                    message TEXT NOT NULL,
                    severity VARCHAR(10) DEFAULT 'info',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_id VARCHAR(50),
                    status VARCHAR(20) DEFAULT 'pending',
                    metadata TEXT  -- JSON string
                )
            ''')
            
            # Activity log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS activity_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id VARCHAR(50) NOT NULL,
                    activity_type VARCHAR(50) NOT NULL,
                    details TEXT,  -- JSON string
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    ip_address VARCHAR(45),
                    user_agent TEXT
                )
            ''')
            
            # Drowning incidents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS drowning_incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_file VARCHAR(255),
                    analysis_results TEXT,  -- JSON string
                    confidence_score REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    location VARCHAR(100),
                    status VARCHAR(20) DEFAULT 'detected'
                )
            ''')
            
            # Suggestions feedback table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS suggestions_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id VARCHAR(50) NOT NULL,
                    suggestion_type VARCHAR(50),
                    suggestion_text TEXT,
                    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
                    feedback_text TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
    
    def insert_environmental_data(self, temperature, ph, flow, emotion=None, location=None, source='manual'):
        """Insert environmental data record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO environmental_data (temperature, ph, flow, emotion, location, source)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (temperature, ph, flow, emotion, location, source))
            return cursor.lastrowid
    
    def get_environmental_data(self, limit=100, start_date=None, end_date=None):
        """Get environmental data records"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = 'SELECT * FROM environmental_data'
            params = []
            
            if start_date or end_date:
                query += ' WHERE'
                if start_date:
                    query += ' timestamp >= ?'
                    params.append(start_date)
                if end_date:
                    if start_date:
                        query += ' AND'
                    query += ' timestamp <= ?'
                    params.append(end_date)
            
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def upsert_user_profile(self, user_id, preferences):
        """Insert or update user profile"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            preferences_json = json.dumps(preferences)
            
            cursor.execute('''
                INSERT OR REPLACE INTO user_profiles (user_id, preferences, last_active)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (user_id, preferences_json))
    
    def get_user_profile(self, user_id):
        """Get user profile"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM user_profiles WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()
            
            if row:
                profile = dict(row)
                profile['preferences'] = json.loads(profile['preferences'])
                return profile
            return None
    
    def insert_alert(self, alert_type, message, severity='info', user_id=None, metadata=None):
        """Insert alert record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor.execute('''
                INSERT INTO alerts (alert_type, message, severity, user_id, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (alert_type, message, severity, user_id, metadata_json))
            return cursor.lastrowid
    
    def get_alerts(self, user_id=None, status='pending', limit=50):
        """Get alerts"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if user_id:
                cursor.execute('''
                    SELECT * FROM alerts 
                    WHERE (user_id = ? OR user_id IS NULL) AND status = ?
                    ORDER BY timestamp DESC LIMIT ?
                ''', (user_id, status, limit))
            else:
                cursor.execute('''
                    SELECT * FROM alerts WHERE status = ?
                    ORDER BY timestamp DESC LIMIT ?
                ''', (status, limit))
            
            alerts = []
            for row in cursor.fetchall():
                alert = dict(row)
                if alert['metadata']:
                    alert['metadata'] = json.loads(alert['metadata'])
                alerts.append(alert)
            return alerts
    
    def log_activity(self, user_id, activity_type, details=None, ip_address=None, user_agent=None):
        """Log user activity"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            details_json = json.dumps(details) if details else None
            
            cursor.execute('''
                INSERT INTO activity_log (user_id, activity_type, details, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, activity_type, details_json, ip_address, user_agent))
    
    def insert_drowning_incident(self, video_file, analysis_results, confidence_score, location=None):
        """Insert drowning incident record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            analysis_json = json.dumps(analysis_results)
            
            cursor.execute('''
                INSERT INTO drowning_incidents (video_file, analysis_results, confidence_score, location)
                VALUES (?, ?, ?, ?)
            ''', (video_file, analysis_json, confidence_score, location))
            return cursor.lastrowid
    
    def insert_suggestion_feedback(self, user_id, suggestion_type, suggestion_text, rating, feedback_text=None):
        """Insert suggestion feedback"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO suggestions_feedback 
                (user_id, suggestion_type, suggestion_text, rating, feedback_text)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, suggestion_type, suggestion_text, rating, feedback_text))
    
    def get_suggestion_feedback_stats(self, suggestion_type=None):
        """Get suggestion feedback statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if suggestion_type:
                cursor.execute('''
                    SELECT AVG(rating) as avg_rating, COUNT(*) as total_feedback
                    FROM suggestions_feedback WHERE suggestion_type = ?
                ''', (suggestion_type,))
            else:
                cursor.execute('''
                    SELECT suggestion_type, AVG(rating) as avg_rating, COUNT(*) as total_feedback
                    FROM suggestions_feedback GROUP BY suggestion_type
                ''')
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_environmental_trends(self, days=7):
        """Get environmental data trends"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    DATE(timestamp) as date,
                    AVG(temperature) as avg_temp,
                    AVG(ph) as avg_ph,
                    AVG(flow) as avg_flow,
                    COUNT(*) as measurements
                FROM environmental_data 
                WHERE timestamp >= datetime('now', '-' || ? || ' days')
                GROUP BY DATE(timestamp)
                ORDER BY date
            ''', (days,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def cleanup_old_data(self, days=30):
        """Clean up old data to prevent database from growing too large"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Clean old activity logs
            cursor.execute('''
                DELETE FROM activity_log 
                WHERE timestamp < datetime('now', '-' || ? || ' days')
            ''', (days,))
            
            # Clean old environmental data (keep important records)
            cursor.execute('''
                DELETE FROM environmental_data 
                WHERE timestamp < datetime('now', '-' || ? || ' days')
                AND emotion NOT IN ('angry', 'sad')  -- Keep concerning records longer
            ''', (days,))

# Global database instance
db = DatabaseManager()
