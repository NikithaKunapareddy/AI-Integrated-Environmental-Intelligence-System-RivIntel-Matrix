# RiverMind API Documentation

## Overview
The RiverMind API provides endpoints for environmental monitoring, drowning detection, emotion analysis, and personalized suggestions for river safety and conservation.

## Base URL
```
http://localhost:5000
```

## Authentication
Currently, the API uses simple user ID-based identification. In production, implement proper authentication (JWT, OAuth, etc.).

## Endpoints

### 1. Environmental Data

#### POST /api/environmental-data
Submit environmental data for analysis.

**Request Body:**
```json
{
    "temperature": 25.5,
    "ph": 7.2,
    "flow": 65.0,
    "location": "River Point A",
    "user_id": "user123"
}
```

**Response:**
```json
{
    "status": "success",
    "data": {
        "id": 1,
        "emotion": "happy",
        "timestamp": "2024-01-01T12:00:00"
    }
}
```

#### GET /api/environmental-data
Retrieve environmental data history.

**Query Parameters:**
- `limit`: Number of records (default: 100)
- `start_date`: Start date (ISO format)
- `end_date`: End date (ISO format)

**Response:**
```json
{
    "status": "success",
    "data": [
        {
            "id": 1,
            "temperature": 25.5,
            "ph": 7.2,
            "flow": 65.0,
            "emotion": "happy",
            "timestamp": "2024-01-01T12:00:00"
        }
    ]
}
```

### 2. Suggestions

#### GET /api/suggestions/{user_id}
Get personalized suggestions for a user.

**Response:**
```json
{
    "status": "success",
    "suggestions": {
        "safety": "Always swim with a buddy",
        "conservation": "Participate in river clean-up events",
        "emotional": "Practice mindfulness by the river",
        "environmental": "Conditions are favorable for river activities"
    },
    "reasoning": "Based on sunny weather, summer season, and your moderate activity level",
    "timestamp": "2024-01-01T12:00:00"
}
```

#### POST /api/suggestions/feedback
Submit feedback on suggestions.

**Request Body:**
```json
{
    "user_id": "user123",
    "suggestion_type": "safety",
    "suggestion_text": "Always swim with a buddy",
    "rating": 5,
    "feedback_text": "Very helpful advice!"
}
```

### 3. Drowning Detection

#### POST /api/drowning-detection
Upload video for drowning detection analysis.

**Request:**
- Multipart form data with video file
- Field name: `video`
- Additional fields: `user_id`, `location`

**Response:**
```json
{
    "status": "success",
    "analysis": {
        "incidents_detected": 2,
        "confidence_score": 0.85,
        "alerts": [
            {
                "timestamp": 15.5,
                "description": "Potential drowning behavior detected",
                "confidence": 0.9
            }
        ]
    }
}
```

### 4. Alerts

#### GET /api/alerts/{user_id}
Get alerts for a specific user.

**Response:**
```json
{
    "status": "success",
    "alerts": [
        {
            "id": 1,
            "alert_type": "safety",
            "message": "High flow detected in your area",
            "severity": "warning",
            "timestamp": "2024-01-01T12:00:00"
        }
    ]
}
```

#### POST /api/alerts
Create a new alert.

**Request Body:**
```json
{
    "alert_type": "emergency",
    "message": "Flood warning in effect",
    "severity": "critical",
    "user_id": "user123",
    "metadata": {
        "location": "River Point A",
        "estimated_duration": "2 hours"
    }
}
```

### 5. River Emotion

#### GET /api/river-emotion
Get current river emotion state.

**Query Parameters:**
- `location`: River location (optional)

**Response:**
```json
{
    "status": "success",
    "emotion": "happy",
    "confidence": 0.85,
    "factors": {
        "temperature": "optimal",
        "ph": "normal",
        "flow": "moderate"
    },
    "timestamp": "2024-01-01T12:00:00"
}
```

### 6. System Health

#### GET /api/health
Check system health status.

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2024-01-01T12:00:00",
    "components": {
        "database": {"status": "healthy"},
        "disk": {
            "status": "healthy",
            "free_space_gb": 25.5
        },
        "uploads_dir": {"status": "healthy"}
    }
}
```

### 7. Analytics

#### GET /api/analytics/trends
Get environmental trends and analytics.

**Query Parameters:**
- `days`: Number of days to analyze (default: 7)

**Response:**
```json
{
    "status": "success",
    "trends": {
        "temperature": {
            "average": 24.5,
            "trend": "increasing",
            "change_percent": 5.2
        },
        "ph": {
            "average": 7.1,
            "trend": "stable",
            "change_percent": 0.1
        },
        "flow": {
            "average": 58.3,
            "trend": "decreasing",
            "change_percent": -3.1
        }
    },
    "daily_data": [
        {
            "date": "2024-01-01",
            "avg_temp": 24.1,
            "avg_ph": 7.0,
            "avg_flow": 60.0
        }
    ]
}
```

## Error Responses

All endpoints return errors in the following format:

```json
{
    "error": "Error type",
    "message": "Detailed error message",
    "timestamp": "2024-01-01T12:00:00"
}
```

### Common HTTP Status Codes:
- `200`: Success
- `400`: Bad Request (validation errors)
- `404`: Not Found
- `429`: Too Many Requests (rate limited)
- `500`: Internal Server Error

## Rate Limiting

- Default: 60 requests per minute per user
- Upload endpoints: 10 requests per minute per user
- Emergency endpoints: No rate limiting

## Data Validation

### Environmental Data:
- Temperature: 0-50°C
- pH: 0-14
- Flow: 0-100 (percentage scale)

### User ID:
- 3-50 characters
- Alphanumeric and underscores only

### Video Files:
- Max size: 100MB
- Formats: mp4, avi, mov, mkv, wmv

## Webhooks (Future Feature)

For real-time notifications, the system will support webhooks:

```json
{
    "event": "emergency_alert",
    "data": {
        "alert_type": "flood",
        "severity": "critical",
        "location": "River Point A"
    },
    "timestamp": "2024-01-01T12:00:00"
}
```

## SDKs and Examples

### Python Example:
```python
import requests

# Submit environmental data
response = requests.post('http://localhost:5000/api/environmental-data', json={
    'temperature': 25.5,
    'ph': 7.2,
    'flow': 65.0,
    'user_id': 'user123'
})

print(response.json())

# Get suggestions
response = requests.get('http://localhost:5000/api/suggestions/user123')
print(response.json())
```

### JavaScript Example:
```javascript
// Submit environmental data
fetch('/api/environmental-data', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        temperature: 25.5,
        ph: 7.2,
        flow: 65.0,
        user_id: 'user123'
    })
})
.then(response => response.json())
.then(data => console.log(data));
```

## Testing

Use the provided test scripts to validate API functionality:

```bash
# Run all tests
python test_api.py

# Run specific test
python test_api.py TestEnvironmentalData
```
