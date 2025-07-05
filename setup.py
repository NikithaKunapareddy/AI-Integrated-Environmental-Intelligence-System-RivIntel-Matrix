#!/usr/bin/env python3
"""
RiverMind Setup Script
Automates the setup process for the RiverMind system.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected")

def install_requirements():
    """Install required Python packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ All packages installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    directories = [
        "uploads",
        "logs",
        "data",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def setup_environment():
    """Setup environment variables"""
    env_example = Path('.env.example')
    env_file = Path('.env')
    
    if env_example.exists() and not env_file.exists():
        shutil.copy(env_example, env_file)
        print("📄 Created .env file from template")
        print("⚠️  Please edit .env file with your actual configuration values")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("⚠️  No .env.example file found")

def initialize_database():
    """Initialize the database"""
    print("🗄️  Initializing database...")
    try:
        from app.database import db
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing database: {e}")

def check_system_requirements():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    
    # Check OpenCV
    try:
        import cv2
        print("✅ OpenCV is available")
    except ImportError:
        print("❌ OpenCV not found")
        return False
    
    # Check Flask
    try:
        import flask
        print("✅ Flask is available")
    except ImportError:
        print("❌ Flask not found")
        return False
    
    # Check other critical packages
    critical_packages = ['numpy', 'pandas', 'matplotlib']
    for package in critical_packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} not found")
            return False
    
    return True

def create_sample_data():
    """Create sample data for testing"""
    print("📊 Creating sample data...")
    try:
        from app.database import db
        import random
        from datetime import datetime, timedelta
        
        # Create sample environmental data
        for i in range(20):
            temp = random.uniform(15, 35)
            ph = random.uniform(6.0, 8.5)
            flow = random.uniform(20, 90)
            
            # Simulate data from past week
            timestamp = datetime.now() - timedelta(days=random.randint(0, 7))
            
            db.insert_environmental_data(
                temperature=temp,
                ph=ph,
                flow=flow,
                emotion="happy" if 20 <= temp <= 30 and 6.5 <= ph <= 8.5 else "neutral",
                location=f"Test Location {i % 3 + 1}",
                source="setup_script"
            )
        
        print("✅ Sample data created successfully")
    except Exception as e:
        print(f"⚠️  Could not create sample data: {e}")

def run_health_check():
    """Run a basic health check"""
    print("🏥 Running health check...")
    try:
        from app.utils import check_system_health
        health = check_system_health()
        
        if health['status'] == 'healthy':
            print("✅ System health check passed")
        else:
            print("⚠️  System health check found issues")
            print(health)
    except Exception as e:
        print(f"❌ Health check failed: {e}")

def main():
    """Main setup function"""
    print("🌊 RiverMind Setup Script")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    install_requirements()
    
    # Create directories
    create_directories()
    
    # Setup environment
    setup_environment()
    
    # Check system requirements
    if not check_system_requirements():
        print("❌ System requirements not met. Please install missing packages.")
        sys.exit(1)
    
    # Initialize database
    initialize_database()
    
    # Create sample data
    create_sample_data()
    
    # Run health check
    run_health_check()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your configuration")
    print("2. Run the application: python main.py")
    print("3. Visit http://localhost:5000 in your browser")
    print("\nFor API documentation, see: API_DOCUMENTATION.md")

if __name__ == "__main__":
    main()
