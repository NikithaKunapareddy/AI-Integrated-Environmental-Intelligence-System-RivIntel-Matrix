import random
from datetime import datetime
import json
import os

# Enhanced suggestions database with weather and season considerations
SUGGESTIONS_DB = {
    'safety': {
        'general': [
            "Always swim with a buddy",
            "Check weather conditions before visiting",
            "Stay within designated swimming areas",
            "Learn basic water rescue techniques",
            "Keep emergency numbers handy"
        ],
        'rainy': [
            "Avoid river activities during heavy rain",
            "Check flood warnings before visiting",
            "Stay away from fast-flowing water",
            "Monitor water levels continuously"
        ],
        'sunny': [
            "Use sunscreen and stay hydrated",
            "Bring shade or umbrella",
            "Check water temperature before swimming",
            "Be aware of increased visitor traffic"
        ]
    },
    'conservation': {
        'general': [
            "Participate in river clean-up events",
            "Reduce plastic usage near water bodies",
            "Report pollution incidents",
            "Support local conservation efforts",
            "Educate others about river protection"
        ],
        'seasonal': [
            "Plant native trees along riverbanks",
            "Monitor water quality changes",
            "Create wildlife corridors",
            "Advocate for reduced industrial discharge"
        ]
    },
    'emotional': {
        'calm': [
            "Practice mindfulness by the river",
            "Keep a river journal",
            "Take time to appreciate nature",
            "Try river meditation techniques"
        ],
        'stressed': [
            "Join river conservation groups",
            "Share your river experiences",
            "Connect with local environmental groups",
            "Consider river therapy sessions"
        ]
    },
    'health': [
        "Monitor air quality near rivers",
        "Check water quality before contact",
        "Be aware of waterborne diseases",
        "Use appropriate protective gear",
        "Stay updated on health advisories"
    ]
}

# User preference tracking (in real app, this would be in a database)
USER_PREFERENCES = {}

def get_weather_condition():
    """
    Simulate weather condition detection.
    In a real app, this would integrate with a weather API.
    """
    conditions = ['sunny', 'rainy', 'cloudy', 'windy']
    return random.choice(conditions)

def get_season():
    """Get current season based on date"""
    month = datetime.now().month
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'

def get_user_profile(user_id):
    """
    Get or create user profile with preferences.
    In a real app, this would connect to a user database.
    """
    if user_id not in USER_PREFERENCES:
        USER_PREFERENCES[user_id] = {
            'interests': ['safety', 'conservation'],
            'activity_level': 'moderate',
            'experience_level': 'beginner',
            'preferred_time': 'morning',
            'last_visit': None
        }
    return USER_PREFERENCES[user_id]

def update_user_feedback(user_id, suggestion_type, rating):
    """
    Update user preferences based on feedback.
    This would help ML models learn user preferences.
    """
    profile = get_user_profile(user_id)
    if 'feedback' not in profile:
        profile['feedback'] = {}
    
    if suggestion_type not in profile['feedback']:
        profile['feedback'][suggestion_type] = []
    
    profile['feedback'][suggestion_type].append({
        'rating': rating,
        'timestamp': datetime.now().isoformat()
    })

def get_personalized_suggestions(user_id, river_emotion=None, environmental_data=None):
    """
    Generate personalized suggestions based on user context and river conditions.
    
    Args:
        user_id (str): Unique user identifier
        river_emotion (str): Current river emotion state
        environmental_data (dict): Current environmental parameters
        
    Returns:
        dict: Personalized suggestions with reasoning
    """
    profile = get_user_profile(user_id)
    weather = get_weather_condition()
    season = get_season()
    hour = datetime.now().hour
    time_of_day = 'morning' if 5 <= hour < 12 else 'afternoon' if 12 <= hour < 17 else 'evening'
    
    suggestions = {}
    
    # Safety suggestions based on weather and conditions
    if weather == 'rainy':
        suggestions['safety'] = random.choice(SUGGESTIONS_DB['safety']['rainy'])
    else:
        suggestions['safety'] = random.choice(SUGGESTIONS_DB['safety']['sunny'])
    
    # Conservation suggestions
    suggestions['conservation'] = random.choice(SUGGESTIONS_DB['conservation']['general'])
    
    # Emotional suggestions based on river emotion
    if river_emotion in ['sad', 'angry', 'stressed']:
        suggestions['emotional'] = random.choice(SUGGESTIONS_DB['emotional']['stressed'])
    else:
        suggestions['emotional'] = random.choice(SUGGESTIONS_DB['emotional']['calm'])
    
    # Health suggestions
    suggestions['health'] = random.choice(SUGGESTIONS_DB['health'])
    
    # Time-specific suggestions
    activity_suggestions = {
        'morning': f"Perfect {time_of_day} for a peaceful riverside walk",
        'afternoon': f"Great {time_of_day} for river monitoring activities",
        'evening': f"Ideal {time_of_day} for reflection and river observation"
    }
    
    # Environmental condition-based suggestions
    if environmental_data:
        temp = environmental_data.get('temperature', 25)
        ph = environmental_data.get('ph', 7)
        flow = environmental_data.get('flow', 50)
        
        if temp > 30:
            suggestions['environmental'] = "High temperature detected - stay hydrated and seek shade"
        elif temp < 15:
            suggestions['environmental'] = "Cool conditions - dress warmly near the river"
        elif ph < 6.5 or ph > 8.5:
            suggestions['environmental'] = "Water quality concerns - avoid direct contact"
        elif flow > 80:
            suggestions['environmental'] = "High flow detected - exercise extreme caution"
        else:
            suggestions['environmental'] = "Conditions are favorable for river activities"
    
    return {
        'suggestions': suggestions,
        'time_specific': activity_suggestions.get(time_of_day, "Enjoy your time by the river"),
        'weather_condition': weather,
        'season': season,
        'river_emotion': river_emotion,
        'user_profile': profile['interests'],
        'timestamp': datetime.now().isoformat(),
        'reasoning': f"Based on {weather} weather, {season} season, and your {profile['activity_level']} activity level"
    }

# Legacy function for backward compatibility
def get_suggestions(user_id):
    """
    Generate personalized suggestions based on user context.
    
    Args:
        user_id (str): Unique user identifier
        
    Returns:
        dict: Personalized suggestions
    """
    return get_personalized_suggestions(user_id)

def get_emergency_suggestions(emergency_type):
    """
    Get emergency-specific suggestions.
    
    Args:
        emergency_type (str): Type of emergency ('flood', 'drought', 'pollution', 'drowning')
        
    Returns:
        list: Emergency-specific suggestions
    """
    emergency_suggestions = {
        'flood': [
            "Move to higher ground immediately",
            "Avoid walking or driving through flood water",
            "Monitor emergency broadcasts",
            "Prepare emergency supplies"
        ],
        'drought': [
            "Conserve water usage",
            "Monitor river levels closely",
            "Report dried river sections",
            "Support water conservation efforts"
        ],
        'pollution': [
            "Avoid direct contact with contaminated water",
            "Report pollution sources to authorities",
            "Document pollution incidents with photos",
            "Support cleanup initiatives"
        ],
        'drowning': [
            "Call emergency services immediately",
            "Throw flotation device if available",
            "Do not enter water unless trained",
            "Provide clear location to responders"
        ]
    }
    
    return emergency_suggestions.get(emergency_type, ["Contact local emergency services"])

def log_user_activity(user_id, activity_type, details=None):
    """
    Log user activity for ML training and personalization.
    
    Args:
        user_id (str): User identifier
        activity_type (str): Type of activity
        details (dict): Additional activity details
    """
    profile = get_user_profile(user_id)
    
    if 'activity_log' not in profile:
        profile['activity_log'] = []
    
    profile['activity_log'].append({
        'activity_type': activity_type,
        'details': details or {},
        'timestamp': datetime.now().isoformat()
    })
    
    # Keep only last 100 activities to prevent unlimited growth
    if len(profile['activity_log']) > 100:
        profile['activity_log'] = profile['activity_log'][-100:]