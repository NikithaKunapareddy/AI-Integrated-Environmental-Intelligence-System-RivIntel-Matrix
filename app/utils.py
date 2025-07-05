"""
Utility functions for error handling, validation, and logging
"""
import logging
import os
from datetime import datetime
from functools import wraps
from flask import jsonify
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rivermind.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def validate_environmental_data(data):
    """
    Validate environmental data parameters.
    
    Args:
        data (dict): Environmental data to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    required_fields = ['temperature', 'ph', 'flow']
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Validate ranges
    if not (0 <= data['temperature'] <= 50):
        return False, "Temperature must be between 0 and 50°C"
    
    if not (0 <= data['ph'] <= 14):
        return False, "pH must be between 0 and 14"
    
    if not (0 <= data['flow'] <= 100):
        return False, "Flow must be between 0 and 100"
    
    return True, None

def validate_user_id(user_id):
    """
    Validate user ID format.
    
    Args:
        user_id (str): User identifier
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not user_id or not isinstance(user_id, str):
        return False
    
    if len(user_id) < 3 or len(user_id) > 50:
        return False
    
    return True

def handle_errors(f):
    """
    Decorator for handling errors in Flask routes.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            
            return jsonify({
                'error': 'Internal server error',
                'message': 'An unexpected error occurred',
                'timestamp': datetime.now().isoformat()
            }), 500
    
    return decorated_function

def validate_video_file(file):
    """
    Validate uploaded video file.
    
    Args:
        file: Flask uploaded file object
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not file:
        return False, "No file uploaded"
    
    if file.filename == '':
        return False, "No file selected"
    
    # Check file extension
    allowed_extensions = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}
    if '.' not in file.filename:
        return False, "File must have an extension"
    
    extension = file.filename.rsplit('.', 1)[1].lower()
    if extension not in allowed_extensions:
        return False, f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
    
    # Check file size (limit to 100MB)
    if file.content_length and file.content_length > 100 * 1024 * 1024:
        return False, "File too large. Maximum size is 100MB"
    
    return True, None

def sanitize_filename(filename):
    """
    Sanitize filename for safe storage.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Sanitized filename
    """
    import re
    
    # Remove path components
    filename = os.path.basename(filename)
    
    # Remove or replace unsafe characters
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    
    # Add timestamp to prevent conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(filename)
    
    return f"{name}_{timestamp}{ext}"

def log_activity(user_id, activity_type, details=None):
    """
    Log user activity for monitoring and analytics.
    
    Args:
        user_id (str): User identifier
        activity_type (str): Type of activity
        details (dict): Additional details
    """
    logger.info(f"User {user_id} performed {activity_type}: {details}")

def check_system_health():
    """
    Check system health and return status.
    
    Returns:
        dict: System health status
    """
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'components': {}
    }
    
    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage('.')
    health_status['components']['disk'] = {
        'status': 'healthy' if free > 1024**3 else 'warning',  # 1GB minimum
        'free_space_gb': free / (1024**3),
        'total_space_gb': total / (1024**3)
    }
    
    # Check uploads directory
    uploads_dir = 'uploads'
    if not os.path.exists(uploads_dir):
        try:
            os.makedirs(uploads_dir)
            health_status['components']['uploads_dir'] = {'status': 'created'}
        except Exception as e:
            health_status['components']['uploads_dir'] = {'status': 'error', 'error': str(e)}
    else:
        health_status['components']['uploads_dir'] = {'status': 'healthy'}
    
    return health_status

def rate_limit_check(user_id, action, max_requests=10, time_window=60):
    """
    Simple rate limiting check.
    
    Args:
        user_id (str): User identifier
        action (str): Action being performed
        max_requests (int): Maximum requests allowed
        time_window (int): Time window in seconds
        
    Returns:
        tuple: (is_allowed, remaining_requests)
    """
    # This is a simplified version - in production, use Redis or similar
    import time
    
    current_time = time.time()
    rate_limit_key = f"{user_id}_{action}"
    
    # For demo purposes, we'll use a simple in-memory store
    if not hasattr(rate_limit_check, 'requests'):
        rate_limit_check.requests = {}
    
    if rate_limit_key not in rate_limit_check.requests:
        rate_limit_check.requests[rate_limit_key] = []
    
    # Clean old requests
    rate_limit_check.requests[rate_limit_key] = [
        req_time for req_time in rate_limit_check.requests[rate_limit_key]
        if current_time - req_time < time_window
    ]
    
    # Check if limit exceeded
    if len(rate_limit_check.requests[rate_limit_key]) >= max_requests:
        return False, 0
    
    # Add current request
    rate_limit_check.requests[rate_limit_key].append(current_time)
    
    remaining = max_requests - len(rate_limit_check.requests[rate_limit_key])
    return True, remaining
