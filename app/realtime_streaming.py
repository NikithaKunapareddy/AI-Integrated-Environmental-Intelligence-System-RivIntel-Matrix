"""
Advanced Real-time Data Streaming and Processing Engine
RivIntel Matrix - Environmental Intelligence System
Author: Nikitha Kunapareddy

This module provides enterprise-grade real-time data streaming capabilities
for environmental monitoring with advanced processing pipelines.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import deque
import websockets
import redis
from kafka import KafkaProducer, KafkaConsumer
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import aiohttp
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty
import pickle
import hashlib
import uuid


@dataclass
class StreamMessage:
    """Represents a streaming data message"""
    id: str
    timestamp: datetime
    source: str
    data_type: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def serialize(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class RealTimeDataStreamer:
    """
    Advanced real-time data streaming engine for environmental monitoring
    Handles high-throughput data ingestion, processing, and distribution
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.connected_clients = set()
        self.data_buffer = deque(maxlen=10000)
        self.processing_queues = {}
        self.metrics = {
            'messages_processed': 0,
            'bytes_processed': 0,
            'errors': 0,
            'active_connections': 0
        }
        
        # Initialize components
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            decode_responses=True
        )
        
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=config.get('kafka_servers', ['localhost:9092']),
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
        )
        
        self.db_engine = create_engine(config.get('database_url', 'sqlite:///rivintel.db'))
        self.Session = sessionmaker(bind=self.db_engine)
        
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 10))
        self.processing_threads = []
        
    async def start_streaming(self):
        """Start the real-time streaming service"""
        self.is_running = True
        self.logger.info("Starting RealTime Data Streaming Engine")
        
        # Start processing threads
        await self.start_processing_threads()
        
        # Start WebSocket server
        await self.start_websocket_server()
        
        # Start Kafka consumers
        await self.start_kafka_consumers()
        
        # Start periodic tasks
        asyncio.create_task(self.periodic_health_check())
        asyncio.create_task(self.periodic_metrics_update())
        
    async def start_processing_threads(self):
        """Start background processing threads"""
        for i in range(self.config.get('processing_threads', 4)):
            thread = threading.Thread(
                target=self.data_processing_worker,
                args=(f"worker_{i}",)
            )
            thread.start()
            self.processing_threads.append(thread)
            
    async def start_websocket_server(self):
        """Start WebSocket server for real-time client connections"""
        async def handle_client(websocket, path):
            self.connected_clients.add(websocket)
            self.metrics['active_connections'] = len(self.connected_clients)
            
            try:
                await websocket.send(json.dumps({
                    'type': 'connection_established',
                    'timestamp': datetime.now().isoformat(),
                    'client_id': str(uuid.uuid4())
                }))
                
                async for message in websocket:
                    await self.handle_client_message(websocket, message)
                    
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.connected_clients.discard(websocket)
                self.metrics['active_connections'] = len(self.connected_clients)
        
        start_server = websockets.serve(
            handle_client,
            self.config.get('websocket_host', 'localhost'),
            self.config.get('websocket_port', 8765)
        )
        
        asyncio.create_task(start_server)
        
    async def start_kafka_consumers(self):
        """Start Kafka consumers for different data streams"""
        consumer_configs = self.config.get('kafka_consumers', {})
        
        for topic, config in consumer_configs.items():
            asyncio.create_task(self.kafka_consumer_worker(topic, config))
            
    async def kafka_consumer_worker(self, topic: str, config: Dict):
        """Kafka consumer worker for specific topic"""
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=config.get('servers', ['localhost:9092']),
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id=config.get('group_id', 'rivintel_group')
        )
        
        for message in consumer:
            try:
                stream_message = StreamMessage(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    source=f"kafka_{topic}",
                    data_type=topic,
                    payload=message.value,
                    metadata={'partition': message.partition, 'offset': message.offset}
                )
                
                await self.process_stream_message(stream_message)
                
            except Exception as e:
                self.logger.error(f"Error processing Kafka message: {e}")
                self.metrics['errors'] += 1
                
    async def process_stream_message(self, message: StreamMessage):
        """Process incoming stream message"""
        try:
            # Add to buffer
            self.data_buffer.append(message)
            
            # Update metrics
            self.metrics['messages_processed'] += 1
            self.metrics['bytes_processed'] += len(message.serialize())
            
            # Route to appropriate processing queue
            queue_key = f"{message.source}_{message.data_type}"
            if queue_key not in self.processing_queues:
                self.processing_queues[queue_key] = Queue()
                
            self.processing_queues[queue_key].put(message)
            
            # Store in Redis for caching
            await self.cache_message(message)
            
            # Broadcast to connected clients
            await self.broadcast_to_clients(message)
            
            # Store in database if configured
            if self.config.get('store_to_db', True):
                await self.store_to_database(message)
                
        except Exception as e:
            self.logger.error(f"Error processing stream message: {e}")
            self.metrics['errors'] += 1
            
    async def cache_message(self, message: StreamMessage):
        """Cache message in Redis"""
        try:
            cache_key = f"stream:{message.source}:{message.data_type}:{message.id}"
            self.redis_client.setex(
                cache_key,
                timedelta(hours=24),
                message.serialize()
            )
            
            # Update latest message for each source
            latest_key = f"latest:{message.source}:{message.data_type}"
            self.redis_client.set(latest_key, message.serialize())
            
        except Exception as e:
            self.logger.error(f"Error caching message: {e}")
            
    async def broadcast_to_clients(self, message: StreamMessage):
        """Broadcast message to all connected WebSocket clients"""
        if not self.connected_clients:
            return
            
        broadcast_data = {
            'type': 'stream_update',
            'timestamp': datetime.now().isoformat(),
            'data': message.to_dict()
        }
        
        disconnected_clients = set()
        
        for client in self.connected_clients:
            try:
                await client.send(json.dumps(broadcast_data, default=str))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
                
        # Remove disconnected clients
        self.connected_clients -= disconnected_clients
        self.metrics['active_connections'] = len(self.connected_clients)
        
    async def store_to_database(self, message: StreamMessage):
        """Store message in database"""
        try:
            session = self.Session()
            
            # Convert to database format
            db_record = {
                'id': message.id,
                'timestamp': message.timestamp,
                'source': message.source,
                'data_type': message.data_type,
                'payload': json.dumps(message.payload),
                'metadata': json.dumps(message.metadata)
            }
            
            # Insert into database
            query = text("""
                INSERT INTO stream_messages 
                (id, timestamp, source, data_type, payload, metadata)
                VALUES (:id, :timestamp, :source, :data_type, :payload, :metadata)
            """)
            
            session.execute(query, db_record)
            session.commit()
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error storing to database: {e}")
            
    def data_processing_worker(self, worker_id: str):
        """Background worker for processing data"""
        self.logger.info(f"Starting data processing worker: {worker_id}")
        
        while self.is_running:
            try:
                # Process messages from queues
                for queue_key, queue in self.processing_queues.items():
                    try:
                        message = queue.get(timeout=1)
                        self.process_message_data(message)
                        queue.task_done()
                    except Empty:
                        continue
                        
                time.sleep(0.1)  # Small delay to prevent CPU spinning
                
            except Exception as e:
                self.logger.error(f"Error in processing worker {worker_id}: {e}")
                
    def process_message_data(self, message: StreamMessage):
        """Process individual message data"""
        try:
            # Apply data transformations based on data type
            if message.data_type == 'sensor_data':
                self.process_sensor_data(message)
            elif message.data_type == 'environmental_data':
                self.process_environmental_data(message)
            elif message.data_type == 'alert_data':
                self.process_alert_data(message)
            elif message.data_type == 'prediction_data':
                self.process_prediction_data(message)
                
        except Exception as e:
            self.logger.error(f"Error processing message data: {e}")
            
    def process_sensor_data(self, message: StreamMessage):
        """Process sensor data specifically"""
        payload = message.payload
        
        # Validate sensor data
        if not self.validate_sensor_data(payload):
            self.logger.warning(f"Invalid sensor data received: {message.id}")
            return
            
        # Apply calibration corrections
        corrected_data = self.apply_sensor_calibration(payload)
        
        # Detect anomalies
        anomalies = self.detect_anomalies(corrected_data)
        
        # Store processed data
        processed_message = StreamMessage(
            id=f"processed_{message.id}",
            timestamp=datetime.now(),
            source=f"processed_{message.source}",
            data_type="processed_sensor_data",
            payload=corrected_data,
            metadata={
                'original_id': message.id,
                'anomalies': anomalies,
                'processing_timestamp': datetime.now().isoformat()
            }
        )
        
        # Send to Kafka for further processing
        self.kafka_producer.send('processed_data', processed_message.to_dict())
        
    def validate_sensor_data(self, data: Dict) -> bool:
        """Validate sensor data format and values"""
        required_fields = ['sensor_id', 'timestamp', 'readings']
        
        for field in required_fields:
            if field not in data:
                return False
                
        # Validate readings are numeric
        try:
            readings = data['readings']
            for key, value in readings.items():
                float(value)
        except (ValueError, TypeError):
            return False
            
        return True
        
    def apply_sensor_calibration(self, data: Dict) -> Dict:
        """Apply calibration corrections to sensor data"""
        calibrated_data = data.copy()
        readings = calibrated_data.get('readings', {})
        
        # Apply calibration coefficients
        calibration_config = self.config.get('sensor_calibration', {})
        sensor_id = data.get('sensor_id')
        
        if sensor_id in calibration_config:
            coefficients = calibration_config[sensor_id]
            
            for param, value in readings.items():
                if param in coefficients:
                    # Apply linear calibration: y = mx + b
                    m = coefficients[param].get('slope', 1.0)
                    b = coefficients[param].get('intercept', 0.0)
                    readings[param] = m * float(value) + b
                    
        calibrated_data['readings'] = readings
        return calibrated_data
        
    def detect_anomalies(self, data: Dict) -> List[Dict]:
        """Detect anomalies in sensor data"""
        anomalies = []
        readings = data.get('readings', {})
        
        # Define normal ranges for different parameters
        normal_ranges = {
            'ph': (6.5, 8.5),
            'dissolved_oxygen': (5.0, 14.0),
            'turbidity': (0.0, 100.0),
            'temperature': (0.0, 35.0),
            'conductivity': (0.0, 2000.0)
        }
        
        for param, value in readings.items():
            if param in normal_ranges:
                min_val, max_val = normal_ranges[param]
                if not (min_val <= float(value) <= max_val):
                    anomalies.append({
                        'parameter': param,
                        'value': value,
                        'expected_range': normal_ranges[param],
                        'severity': 'high' if value < min_val * 0.5 or value > max_val * 1.5 else 'medium'
                    })
                    
        return anomalies
        
    async def handle_client_message(self, websocket, message: str):
        """Handle incoming client messages"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'subscribe':
                await self.handle_subscription(websocket, data)
            elif message_type == 'unsubscribe':
                await self.handle_unsubscription(websocket, data)
            elif message_type == 'get_latest':
                await self.handle_get_latest(websocket, data)
            elif message_type == 'get_metrics':
                await self.handle_get_metrics(websocket)
                
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'Invalid JSON format'
            }))
            
    async def handle_subscription(self, websocket, data: Dict):
        """Handle client subscription requests"""
        # Implementation for subscription management
        pass
        
    async def handle_get_latest(self, websocket, data: Dict):
        """Handle requests for latest data"""
        source = data.get('source')
        data_type = data.get('data_type')
        
        if source and data_type:
            latest_key = f"latest:{source}:{data_type}"
            latest_data = self.redis_client.get(latest_key)
            
            if latest_data:
                await websocket.send(json.dumps({
                    'type': 'latest_data',
                    'data': json.loads(latest_data)
                }))
            else:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': 'No data available'
                }))
                
    async def handle_get_metrics(self, websocket):
        """Handle requests for system metrics"""
        metrics_data = {
            'type': 'metrics',
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics.copy()
        }
        
        await websocket.send(json.dumps(metrics_data))
        
    async def periodic_health_check(self):
        """Periodic health check and maintenance"""
        while self.is_running:
            try:
                # Check system health
                health_status = await self.check_system_health()
                
                # Log health status
                self.logger.info(f"System health check: {health_status}")
                
                # Cleanup old data
                await self.cleanup_old_data()
                
                # Wait for next check
                await asyncio.sleep(self.config.get('health_check_interval', 300))
                
            except Exception as e:
                self.logger.error(f"Error in health check: {e}")
                
    async def check_system_health(self) -> Dict:
        """Check overall system health"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'components': {}
        }
        
        # Check Redis connection
        try:
            self.redis_client.ping()
            health_status['components']['redis'] = 'healthy'
        except:
            health_status['components']['redis'] = 'unhealthy'
            health_status['status'] = 'degraded'
            
        # Check database connection
        try:
            session = self.Session()
            session.execute(text("SELECT 1"))
            session.close()
            health_status['components']['database'] = 'healthy'
        except:
            health_status['components']['database'] = 'unhealthy'
            health_status['status'] = 'degraded'
            
        # Check Kafka connection
        try:
            # Simple producer test
            future = self.kafka_producer.send('health_check', {'test': 'data'})
            future.get(timeout=5)
            health_status['components']['kafka'] = 'healthy'
        except:
            health_status['components']['kafka'] = 'unhealthy'
            health_status['status'] = 'degraded'
            
        return health_status
        
    async def cleanup_old_data(self):
        """Cleanup old cached data"""
        try:
            # Remove old cached messages
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            # Get all stream keys
            stream_keys = self.redis_client.keys('stream:*')
            
            for key in stream_keys:
                try:
                    data = self.redis_client.get(key)
                    if data:
                        message_data = json.loads(data)
                        message_time = datetime.fromisoformat(message_data['timestamp'])
                        
                        if message_time < cutoff_time:
                            self.redis_client.delete(key)
                            
                except (json.JSONDecodeError, KeyError, ValueError):
                    # Delete corrupted data
                    self.redis_client.delete(key)
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
            
    async def periodic_metrics_update(self):
        """Periodic metrics update and reporting"""
        while self.is_running:
            try:
                # Update metrics
                self.metrics['timestamp'] = datetime.now().isoformat()
                self.metrics['buffer_size'] = len(self.data_buffer)
                self.metrics['processing_queues'] = len(self.processing_queues)
                
                # Store metrics in Redis
                metrics_key = f"metrics:{datetime.now().strftime('%Y%m%d_%H%M')}"
                self.redis_client.setex(metrics_key, timedelta(hours=24), json.dumps(self.metrics))
                
                # Send metrics to Kafka
                self.kafka_producer.send('system_metrics', self.metrics)
                
                await asyncio.sleep(self.config.get('metrics_update_interval', 60))
                
            except Exception as e:
                self.logger.error(f"Error updating metrics: {e}")
                
    async def stop_streaming(self):
        """Stop the streaming service"""
        self.is_running = False
        self.logger.info("Stopping RealTime Data Streaming Engine")
        
        # Close connections
        for client in self.connected_clients:
            await client.close()
            
        # Close Kafka producer
        self.kafka_producer.close()
        
        # Wait for processing threads to finish
        for thread in self.processing_threads:
            thread.join(timeout=30)
            
        # Close database connections
        self.db_engine.dispose()
        
        # Close Redis connection
        self.redis_client.close()


class StreamingAnalytics:
    """
    Real-time analytics engine for streaming data
    Provides real-time calculations and insights
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.data_windows = {}
        self.analytics_results = {}
        
    def add_data_point(self, stream_id: str, data_point: Dict):
        """Add new data point to analytics window"""
        if stream_id not in self.data_windows:
            self.data_windows[stream_id] = deque(maxlen=self.window_size)
            
        self.data_windows[stream_id].append(data_point)
        
        # Calculate analytics
        self.calculate_analytics(stream_id)
        
    def calculate_analytics(self, stream_id: str):
        """Calculate real-time analytics for stream"""
        if stream_id not in self.data_windows:
            return
            
        data_window = list(self.data_windows[stream_id])
        
        if not data_window:
            return
            
        # Extract numeric values
        numeric_data = self.extract_numeric_values(data_window)
        
        if not numeric_data:
            return
            
        # Calculate statistics
        analytics = {}
        
        for param, values in numeric_data.items():
            if values:
                analytics[param] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'trend': self.calculate_trend(values),
                    'anomaly_score': self.calculate_anomaly_score(values)
                }
                
        self.analytics_results[stream_id] = {
            'timestamp': datetime.now().isoformat(),
            'analytics': analytics,
            'data_points': len(data_window)
        }
        
    def extract_numeric_values(self, data_window: List[Dict]) -> Dict[str, List[float]]:
        """Extract numeric values from data window"""
        numeric_data = {}
        
        for data_point in data_window:
            readings = data_point.get('readings', {})
            
            for param, value in readings.items():
                try:
                    numeric_value = float(value)
                    
                    if param not in numeric_data:
                        numeric_data[param] = []
                        
                    numeric_data[param].append(numeric_value)
                    
                except (ValueError, TypeError):
                    continue
                    
        return numeric_data
        
    def calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return 'stable'
            
        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
            
    def calculate_anomaly_score(self, values: List[float]) -> float:
        """Calculate anomaly score for values"""
        if len(values) < 10:
            return 0.0
            
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return 0.0
            
        # Calculate z-scores
        z_scores = np.abs((np.array(values) - mean) / std)
        
        # Return maximum z-score as anomaly score
        return float(np.max(z_scores))
        
    def get_analytics(self, stream_id: str) -> Optional[Dict]:
        """Get current analytics for stream"""
        return self.analytics_results.get(stream_id)
        
    def get_all_analytics(self) -> Dict:
        """Get analytics for all streams"""
        return self.analytics_results.copy()


# Demo and testing functions
async def demo_streaming_engine():
    """Demonstrate the streaming engine capabilities"""
    config = {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'kafka_servers': ['localhost:9092'],
        'database_url': 'sqlite:///rivintel_streaming.db',
        'websocket_host': 'localhost',
        'websocket_port': 8765,
        'max_workers': 4,
        'processing_threads': 2,
        'health_check_interval': 60,
        'metrics_update_interval': 30
    }
    
    streamer = RealTimeDataStreamer(config)
    
    # Start streaming
    await streamer.start_streaming()
    
    # Simulate some data
    for i in range(100):
        message = StreamMessage(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            source="demo_sensor",
            data_type="sensor_data",
            payload={
                'sensor_id': f'sensor_{i % 10}',
                'readings': {
                    'ph': 7.0 + np.random.normal(0, 0.5),
                    'dissolved_oxygen': 8.0 + np.random.normal(0, 1.0),
                    'temperature': 20.0 + np.random.normal(0, 2.0)
                }
            },
            metadata={'demo': True}
        )
        
        await streamer.process_stream_message(message)
        await asyncio.sleep(0.1)
        
    # Keep running for demo
    await asyncio.sleep(30)
    
    # Stop streaming
    await streamer.stop_streaming()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_streaming_engine())
