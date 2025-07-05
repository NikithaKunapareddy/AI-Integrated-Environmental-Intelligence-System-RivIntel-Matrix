"""
Advanced API Gateway and Microservices Architecture
RivIntel Matrix - Environmental Intelligence System
Author: Nikitha Kunapareddy

This module provides a comprehensive API gateway with microservices architecture
for scalable environmental monitoring and AI-powered analytics.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import uuid
import hashlib
import hmac
import jwt
from functools import wraps
import aiohttp
from aiohttp import web, ClientSession, ClientTimeout
import aioredis
import aiodns
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import structlog
import traceback
from circuit_breaker import CircuitBreaker
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class ServiceEndpoint:
    """Represents a microservice endpoint"""
    service_name: str
    host: str
    port: int
    path: str
    health_check_path: str
    timeout: int = 30
    retries: int = 3
    circuit_breaker_threshold: int = 5
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"
        
    @property
    def full_url(self) -> str:
        return f"{self.base_url}{self.path}"
        
    @property
    def health_url(self) -> str:
        return f"{self.base_url}{self.health_check_path}"


@dataclass
class APIRequest:
    """Represents an API request"""
    request_id: str
    timestamp: datetime
    method: str
    path: str
    headers: Dict[str, str]
    query_params: Dict[str, str]
    body: Optional[Dict[str, Any]]
    user_id: Optional[str]
    client_ip: str
    user_agent: str


class APIGateway:
    """
    Advanced API Gateway with load balancing, authentication, 
    rate limiting, and comprehensive monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = structlog.get_logger(__name__)
        self.app = web.Application(middlewares=[
            self.auth_middleware,
            self.rate_limit_middleware,
            self.logging_middleware,
            self.cors_middleware
        ])
        
        # Service registry
        self.services = {}
        self.circuit_breakers = {}
        self.load_balancers = {}
        
        # Monitoring and metrics
        self.setup_metrics()
        
        # Database and cache
        self.redis = None
        self.db_engine = None
        self.session_maker = None
        
        # HTTP client session
        self.client_session = None
        
        # Setup routes
        self.setup_routes()
        
    def setup_metrics(self):
        """Setup Prometheus metrics"""
        self.metrics = {
            'requests_total': Counter(
                'api_gateway_requests_total',
                'Total API requests',
                ['method', 'endpoint', 'status']
            ),
            'request_duration': Histogram(
                'api_gateway_request_duration_seconds',
                'Request duration in seconds',
                ['method', 'endpoint']
            ),
            'active_connections': Gauge(
                'api_gateway_active_connections',
                'Number of active connections'
            ),
            'service_health': Gauge(
                'api_gateway_service_health',
                'Service health status',
                ['service_name']
            ),
            'rate_limit_hits': Counter(
                'api_gateway_rate_limit_hits_total',
                'Rate limit hits',
                ['user_id', 'endpoint']
            )
        }
        
    async def initialize(self):
        """Initialize the API Gateway"""
        # Initialize Redis
        self.redis = await aioredis.from_url(
            self.config.get('redis_url', 'redis://localhost:6379')
        )
        
        # Initialize database
        self.db_engine = create_async_engine(
            self.config.get('database_url', 'sqlite+aiosqlite:///rivintel.db')
        )
        self.session_maker = sessionmaker(
            self.db_engine, class_=AsyncSession, expire_on_commit=False
        )
        
        # Initialize HTTP client
        timeout = ClientTimeout(total=30)
        self.client_session = ClientSession(timeout=timeout)
        
        # Register services
        await self.register_services()
        
        # Start health checking
        asyncio.create_task(self.health_check_loop())
        
        # Start metrics collection
        asyncio.create_task(self.metrics_collection_loop())
        
    async def register_services(self):
        """Register microservices"""
        service_configs = self.config.get('services', {})
        
        for service_name, config in service_configs.items():
            endpoint = ServiceEndpoint(
                service_name=service_name,
                host=config['host'],
                port=config['port'],
                path=config.get('path', '/'),
                health_check_path=config.get('health_check_path', '/health'),
                timeout=config.get('timeout', 30),
                retries=config.get('retries', 3),
                circuit_breaker_threshold=config.get('circuit_breaker_threshold', 5)
            )
            
            self.services[service_name] = endpoint
            
            # Initialize circuit breaker
            self.circuit_breakers[service_name] = CircuitBreaker(
                failure_threshold=endpoint.circuit_breaker_threshold,
                recovery_timeout=60,
                expected_exception=Exception
            )
            
            # Initialize load balancer (simple round-robin for now)
            self.load_balancers[service_name] = RoundRobinLoadBalancer([endpoint])
            
    def setup_routes(self):
        """Setup API routes"""
        # Health check
        self.app.router.add_get('/health', self.health_check)
        
        # Metrics endpoint
        self.app.router.add_get('/metrics', self.metrics_endpoint)
        
        # Authentication endpoints
        self.app.router.add_post('/auth/login', self.login)
        self.app.router.add_post('/auth/refresh', self.refresh_token)
        self.app.router.add_post('/auth/logout', self.logout)
        
        # API versioning
        self.app.router.add_route('*', '/api/v1/{service}/{path:.*}', self.proxy_request)
        self.app.router.add_route('*', '/api/v2/{service}/{path:.*}', self.proxy_request_v2)
        
        # WebSocket proxy
        self.app.router.add_get('/ws/{service}', self.websocket_proxy)
        
    @web.middleware
    async def auth_middleware(self, request: web.Request, handler: Callable):
        """Authentication middleware"""
        # Skip auth for health check and metrics
        if request.path in ['/health', '/metrics']:
            return await handler(request)
            
        # Skip auth for login endpoints
        if request.path.startswith('/auth/'):
            return await handler(request)
            
        # Extract token
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return web.json_response(
                {'error': 'Missing or invalid authorization header'},
                status=401
            )
            
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        
        try:
            # Verify JWT token
            payload = jwt.decode(
                token,
                self.config.get('jwt_secret', 'secret'),
                algorithms=['HS256']
            )
            
            # Check token expiration
            if payload.get('exp', 0) < time.time():
                return web.json_response(
                    {'error': 'Token expired'},
                    status=401
                )
                
            # Add user info to request
            request['user'] = payload
            
        except jwt.InvalidTokenError:
            return web.json_response(
                {'error': 'Invalid token'},
                status=401
            )
            
        return await handler(request)
        
    @web.middleware
    async def rate_limit_middleware(self, request: web.Request, handler: Callable):
        """Rate limiting middleware"""
        # Skip rate limiting for health check
        if request.path in ['/health', '/metrics']:
            return await handler(request)
            
        # Get user ID or IP for rate limiting
        user_id = request.get('user', {}).get('user_id')
        client_ip = request.remote
        rate_limit_key = user_id or client_ip
        
        # Rate limit configuration
        rate_limit_config = self.config.get('rate_limiting', {})
        requests_per_minute = rate_limit_config.get('requests_per_minute', 100)
        
        # Check rate limit
        redis_key = f"rate_limit:{rate_limit_key}:{int(time.time() // 60)}"
        current_requests = await self.redis.incr(redis_key)
        await self.redis.expire(redis_key, 60)
        
        if current_requests > requests_per_minute:
            self.metrics['rate_limit_hits'].labels(
                user_id=user_id or 'anonymous',
                endpoint=request.path
            ).inc()
            
            return web.json_response(
                {
                    'error': 'Rate limit exceeded',
                    'limit': requests_per_minute,
                    'window': '1 minute'
                },
                status=429
            )
            
        return await handler(request)
        
    @web.middleware
    async def logging_middleware(self, request: web.Request, handler: Callable):
        """Request logging middleware"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Create API request object
        api_request = APIRequest(
            request_id=request_id,
            timestamp=datetime.now(),
            method=request.method,
            path=request.path,
            headers=dict(request.headers),
            query_params=dict(request.query),
            body=await self.get_request_body(request),
            user_id=request.get('user', {}).get('user_id'),
            client_ip=request.remote,
            user_agent=request.headers.get('User-Agent', '')
        )
        
        # Add request ID to request
        request['request_id'] = request_id
        request['api_request'] = api_request
        
        try:
            response = await handler(request)
            status = response.status
        except Exception as e:
            status = 500
            self.logger.error(
                "Request failed",
                request_id=request_id,
                error=str(e),
                traceback=traceback.format_exc()
            )
            raise
        finally:
            duration = time.time() - start_time
            
            # Update metrics
            self.metrics['requests_total'].labels(
                method=request.method,
                endpoint=request.path,
                status=status
            ).inc()
            
            self.metrics['request_duration'].labels(
                method=request.method,
                endpoint=request.path
            ).observe(duration)
            
            # Log request
            self.logger.info(
                "API request completed",
                request_id=request_id,
                method=request.method,
                path=request.path,
                status=status,
                duration=duration,
                user_id=api_request.user_id
            )
            
        return response
        
    @web.middleware
    async def cors_middleware(self, request: web.Request, handler: Callable):
        """CORS middleware"""
        if request.method == 'OPTIONS':
            # Handle preflight request
            return web.Response(
                headers={
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
                    'Access-Control-Max-Age': '86400'
                }
            )
            
        response = await handler(request)
        
        # Add CORS headers
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        
        return response
        
    async def get_request_body(self, request: web.Request) -> Optional[Dict]:
        """Extract request body safely"""
        try:
            if request.content_type == 'application/json':
                return await request.json()
        except:
            pass
        return None
        
    async def proxy_request(self, request: web.Request) -> web.Response:
        """Proxy request to microservice (v1)"""
        service_name = request.match_info['service']
        path = request.match_info.get('path', '')
        
        return await self.forward_request(request, service_name, path, 'v1')
        
    async def proxy_request_v2(self, request: web.Request) -> web.Response:
        """Proxy request to microservice (v2)"""
        service_name = request.match_info['service']
        path = request.match_info.get('path', '')
        
        return await self.forward_request(request, service_name, path, 'v2')
        
    async def forward_request(
        self,
        request: web.Request,
        service_name: str,
        path: str,
        version: str
    ) -> web.Response:
        """Forward request to appropriate microservice"""
        
        if service_name not in self.services:
            return web.json_response(
                {'error': f'Service {service_name} not found'},
                status=404
            )
            
        # Get service endpoint
        load_balancer = self.load_balancers[service_name]
        endpoint = load_balancer.get_next_endpoint()
        
        if not endpoint:
            return web.json_response(
                {'error': f'Service {service_name} unavailable'},
                status=503
            )
            
        # Build target URL
        target_url = f"{endpoint.base_url}/api/{version}/{path}"
        if request.query_string:
            target_url += f"?{request.query_string}"
            
        # Prepare headers
        headers = dict(request.headers)
        headers['X-Request-ID'] = request['request_id']
        headers['X-User-ID'] = str(request.get('user', {}).get('user_id', ''))
        
        # Forward request with circuit breaker
        circuit_breaker = self.circuit_breakers[service_name]
        
        try:
            response = await circuit_breaker.call(
                self._make_http_request,
                request.method,
                target_url,
                headers=headers,
                data=await request.read() if request.has_body else None
            )
            
            # Create response
            return web.Response(
                body=await response.read(),
                status=response.status,
                headers=response.headers
            )
            
        except Exception as e:
            self.logger.error(
                "Service request failed",
                service=service_name,
                endpoint=target_url,
                error=str(e)
            )
            
            return web.json_response(
                {
                    'error': 'Service temporarily unavailable',
                    'service': service_name,
                    'request_id': request['request_id']
                },
                status=503
            )
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _make_http_request(self, method: str, url: str, **kwargs):
        """Make HTTP request with retry logic"""
        async with self.client_session.request(method, url, **kwargs) as response:
            return response
            
    async def websocket_proxy(self, request: web.Request) -> web.WebSocketResponse:
        """WebSocket proxy to microservices"""
        service_name = request.match_info['service']
        
        if service_name not in self.services:
            return web.json_response(
                {'error': f'Service {service_name} not found'},
                status=404
            )
            
        # TODO: Implement WebSocket proxying
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # For now, return a simple message
        await ws.send_str(json.dumps({
            'type': 'connection_established',
            'service': service_name,
            'timestamp': datetime.now().isoformat()
        }))
        
        return ws
        
    async def health_check(self, request: web.Request) -> web.Response:
        """Gateway health check endpoint"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': {},
            'metrics': {
                'active_connections': self.metrics['active_connections']._value.get(),
                'total_requests': sum(
                    family.samples[0].value
                    for family in self.metrics['requests_total'].collect()
                    for sample in family.samples
                )
            }
        }
        
        # Check service health
        for service_name, endpoint in self.services.items():
            try:
                async with self.client_session.get(
                    endpoint.health_url,
                    timeout=ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        health_status['services'][service_name] = 'healthy'
                        self.metrics['service_health'].labels(
                            service_name=service_name
                        ).set(1)
                    else:
                        health_status['services'][service_name] = 'unhealthy'
                        health_status['status'] = 'degraded'
                        self.metrics['service_health'].labels(
                            service_name=service_name
                        ).set(0)
                        
            except Exception as e:
                health_status['services'][service_name] = f'error: {str(e)}'
                health_status['status'] = 'degraded'
                self.metrics['service_health'].labels(
                    service_name=service_name
                ).set(0)
                
        return web.json_response(health_status)
        
    async def metrics_endpoint(self, request: web.Request) -> web.Response:
        """Prometheus metrics endpoint"""
        return web.Response(
            text=generate_latest(),
            content_type='text/plain; version=0.0.4'
        )
        
    async def login(self, request: web.Request) -> web.Response:
        """User login endpoint"""
        try:
            data = await request.json()
            username = data.get('username')
            password = data.get('password')
            
            if not username or not password:
                return web.json_response(
                    {'error': 'Username and password required'},
                    status=400
                )
                
            # TODO: Implement actual authentication
            # For demo, accept any username/password
            user_id = str(uuid.uuid4())
            
            # Generate JWT token
            payload = {
                'user_id': user_id,
                'username': username,
                'exp': int(time.time()) + 3600,  # 1 hour
                'iat': int(time.time())
            }
            
            token = jwt.encode(
                payload,
                self.config.get('jwt_secret', 'secret'),
                algorithm='HS256'
            )
            
            # Generate refresh token
            refresh_payload = {
                'user_id': user_id,
                'type': 'refresh',
                'exp': int(time.time()) + 86400 * 7,  # 7 days
                'iat': int(time.time())
            }
            
            refresh_token = jwt.encode(
                refresh_payload,
                self.config.get('jwt_secret', 'secret'),
                algorithm='HS256'
            )
            
            # Store refresh token in Redis
            await self.redis.setex(
                f"refresh_token:{user_id}",
                86400 * 7,
                refresh_token
            )
            
            return web.json_response({
                'access_token': token,
                'refresh_token': refresh_token,
                'expires_in': 3600,
                'user': {
                    'user_id': user_id,
                    'username': username
                }
            })
            
        except Exception as e:
            return web.json_response(
                {'error': 'Login failed'},
                status=500
            )
            
    async def refresh_token(self, request: web.Request) -> web.Response:
        """Refresh JWT token"""
        try:
            data = await request.json()
            refresh_token = data.get('refresh_token')
            
            if not refresh_token:
                return web.json_response(
                    {'error': 'Refresh token required'},
                    status=400
                )
                
            # Verify refresh token
            payload = jwt.decode(
                refresh_token,
                self.config.get('jwt_secret', 'secret'),
                algorithms=['HS256']
            )
            
            user_id = payload.get('user_id')
            
            # Check if refresh token exists in Redis
            stored_token = await self.redis.get(f"refresh_token:{user_id}")
            if not stored_token or stored_token != refresh_token:
                return web.json_response(
                    {'error': 'Invalid refresh token'},
                    status=401
                )
                
            # Generate new access token
            new_payload = {
                'user_id': user_id,
                'username': payload.get('username'),
                'exp': int(time.time()) + 3600,
                'iat': int(time.time())
            }
            
            new_token = jwt.encode(
                new_payload,
                self.config.get('jwt_secret', 'secret'),
                algorithm='HS256'
            )
            
            return web.json_response({
                'access_token': new_token,
                'expires_in': 3600
            })
            
        except jwt.InvalidTokenError:
            return web.json_response(
                {'error': 'Invalid refresh token'},
                status=401
            )
        except Exception as e:
            return web.json_response(
                {'error': 'Token refresh failed'},
                status=500
            )
            
    async def logout(self, request: web.Request) -> web.Response:
        """User logout endpoint"""
        try:
            user = request.get('user', {})
            user_id = user.get('user_id')
            
            if user_id:
                # Remove refresh token from Redis
                await self.redis.delete(f"refresh_token:{user_id}")
                
            return web.json_response({'message': 'Logged out successfully'})
            
        except Exception as e:
            return web.json_response(
                {'error': 'Logout failed'},
                status=500
            )
            
    async def health_check_loop(self):
        """Periodic health checking of services"""
        while True:
            try:
                for service_name, endpoint in self.services.items():
                    try:
                        async with self.client_session.get(
                            endpoint.health_url,
                            timeout=ClientTimeout(total=5)
                        ) as response:
                            if response.status == 200:
                                self.metrics['service_health'].labels(
                                    service_name=service_name
                                ).set(1)
                            else:
                                self.metrics['service_health'].labels(
                                    service_name=service_name
                                ).set(0)
                                
                    except Exception as e:
                        self.metrics['service_health'].labels(
                            service_name=service_name
                        ).set(0)
                        
                        self.logger.warning(
                            "Service health check failed",
                            service=service_name,
                            error=str(e)
                        )
                        
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error("Health check loop error", error=str(e))
                await asyncio.sleep(60)
                
    async def metrics_collection_loop(self):
        """Periodic metrics collection"""
        while True:
            try:
                # Update active connections metric
                # This would need to be implemented based on actual connection tracking
                
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                self.logger.error("Metrics collection error", error=str(e))
                await asyncio.sleep(60)
                
    async def cleanup(self):
        """Cleanup resources"""
        if self.client_session:
            await self.client_session.close()
            
        if self.redis:
            await self.redis.close()
            
        if self.db_engine:
            await self.db_engine.dispose()


class RoundRobinLoadBalancer:
    """Simple round-robin load balancer"""
    
    def __init__(self, endpoints: List[ServiceEndpoint]):
        self.endpoints = endpoints
        self.current_index = 0
        
    def get_next_endpoint(self) -> Optional[ServiceEndpoint]:
        """Get next endpoint in round-robin fashion"""
        if not self.endpoints:
            return None
            
        endpoint = self.endpoints[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.endpoints)
        
        return endpoint
        
    def remove_endpoint(self, endpoint: ServiceEndpoint):
        """Remove unhealthy endpoint"""
        if endpoint in self.endpoints:
            self.endpoints.remove(endpoint)
            
    def add_endpoint(self, endpoint: ServiceEndpoint):
        """Add healthy endpoint back"""
        if endpoint not in self.endpoints:
            self.endpoints.append(endpoint)


class ServiceRegistry:
    """Service discovery and registration"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.services = {}
        
    async def register_service(
        self,
        service_name: str,
        instance_id: str,
        host: str,
        port: int,
        metadata: Dict[str, Any] = None
    ):
        """Register a service instance"""
        service_key = f"service:{service_name}:{instance_id}"
        service_data = {
            'host': host,
            'port': port,
            'registered_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        await self.redis.setex(
            service_key,
            300,  # 5 minutes TTL
            json.dumps(service_data)
        )
        
    async def discover_services(self, service_name: str) -> List[Dict]:
        """Discover service instances"""
        pattern = f"service:{service_name}:*"
        keys = await self.redis.keys(pattern)
        
        services = []
        for key in keys:
            data = await self.redis.get(key)
            if data:
                services.append(json.loads(data))
                
        return services
        
    async def heartbeat(self, service_name: str, instance_id: str):
        """Update service heartbeat"""
        service_key = f"service:{service_name}:{instance_id}"
        ttl = await self.redis.ttl(service_key)
        
        if ttl > 0:
            await self.redis.expire(service_key, 300)


# Demo and testing functions
async def create_api_gateway():
    """Create and configure API Gateway"""
    config = {
        'redis_url': 'redis://localhost:6379',
        'database_url': 'sqlite+aiosqlite:///rivintel_gateway.db',
        'jwt_secret': 'your-secret-key',
        'rate_limiting': {
            'requests_per_minute': 100
        },
        'services': {
            'water_quality': {
                'host': 'localhost',
                'port': 8001,
                'path': '/api',
                'health_check_path': '/health'
            },
            'predictions': {
                'host': 'localhost',
                'port': 8002,
                'path': '/api',
                'health_check_path': '/health'
            },
            'alerts': {
                'host': 'localhost',
                'port': 8003,
                'path': '/api',
                'health_check_path': '/health'
            }
        }
    }
    
    gateway = APIGateway(config)
    await gateway.initialize()
    
    return gateway


async def run_api_gateway():
    """Run the API Gateway"""
    gateway = await create_api_gateway()
    
    # Start the web application
    runner = web.AppRunner(gateway.app)
    await runner.setup()
    
    site = web.TCPSite(runner, 'localhost', 8000)
    await site.start()
    
    print("API Gateway running on http://localhost:8000")
    
    try:
        # Keep running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await gateway.cleanup()
        await runner.cleanup()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    asyncio.run(run_api_gateway())
