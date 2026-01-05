import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# File paths
TRAINING_DATA_FILE = BASE_DIR / "Dataset/Dataset.csv"
NEW_DATA_FILE = BASE_DIR / "newData.csv"
OUTPUT_FILE = BASE_DIR / "newData_predictions.csv"

# Model files
MODEL_DIR = BASE_DIR / "models"
SCALER_FILE = MODEL_DIR / "scaler.pkl"
MODEL_FILE = MODEL_DIR / "lstm_model.h5"

# Model parameters
TIME_STEPS = 3
LSTM_UNITS = 64
DENSE_UNITS = 32
DROPOUT_RATE = 0.2
BATCH_SIZE = 8
EPOCHS = 50
VALIDATION_SPLIT = 0.2

# Sentiment analysis parameters
SENTIMENT_POSITIVE_WORDS = [
    'excellent', 'outstanding', 'good', 'great', 'positive', 
    'exceptional', 'strong', 'successful', 'effective', 'improving'
]
SENTIMENT_NEGATIVE_WORDS = [
    'poor', 'bad', 'failed', 'missed', 'issues', 'concerns', 
    'struggling', 'problems', 'delays', 'weak'
]

# Risk assessment parameters
RISK_KEYWORDS = {
    'escalation': 0.9, 'critical': 0.85, 'urgent': 0.8, 'immediate': 0.75,
    'intervention': 0.7, 'improvement plan': 0.65, 'review': 0.6,
    'issues': 0.55, 'concerns': 0.5, 'training': 0.4, 'coaching': 0.35,
    'mentoring': 0.3, 'guidance': 0.25, 'supervision': 0.2,
    'continue': 0.15, 'maintain': 0.1, 'leadership': 0.05
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    'EXCELLENT': 90,
    'GOOD': 80,
    'AVERAGE': 70,
    'NEEDS_IMPROVEMENT': 60,
    'POOR': 0
}

# Risk thresholds
RISK_THRESHOLDS = {
    'HIGH': 0.6,
    'MEDIUM': 0.4,
    'LOW': 0.2
}

def setup_directories():
    directories = [MODEL_DIR]
    for directory in directories:
        directory.mkdir(exist_ok=True)
    print(f"Directories created: models/")