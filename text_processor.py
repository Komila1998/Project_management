import numpy as np
from textblob import TextBlob
from config import (
    SENTIMENT_POSITIVE_WORDS, 
    SENTIMENT_NEGATIVE_WORDS,
    RISK_KEYWORDS,
    PERFORMANCE_THRESHOLDS,
    RISK_THRESHOLDS
)

def extract_sentiment_score(feedback_text):
    try:
        blob = TextBlob(str(feedback_text))
        sentiment = blob.sentiment.polarity  # -1 to 1
        normalized = (sentiment + 1) / 2  # Convert to 0-1
        return round(normalized, 2)
    except:
        # Fallback to keyword matching
        return _keyword_based_sentiment(str(feedback_text))

def _keyword_based_sentiment(text):
    text_lower = text.lower()
    
    pos_count = sum(1 for word in SENTIMENT_POSITIVE_WORDS if word in text_lower)
    neg_count = sum(1 for word in SENTIMENT_NEGATIVE_WORDS if word in text_lower)
    
    if pos_count > neg_count:
        return round(0.7 + np.random.uniform(-0.1, 0.1), 2)
    elif neg_count > pos_count:
        return round(0.3 + np.random.uniform(-0.1, 0.1), 2)
    else:
        return round(0.5 + np.random.uniform(-0.1, 0.1), 2)

def calculate_risk_factor(actions_text):
    text = str(actions_text).lower()
    
    risk_score = 0.1  # Base risk
    
    for keyword, weight in RISK_KEYWORDS.items():
        if keyword in text:
            risk_score = max(risk_score, weight)
    
    # Add small variation
    risk_score += np.random.uniform(-0.05, 0.05)
    risk_score = np.clip(risk_score, 0.01, 0.95)
    
    return round(risk_score, 2)

def categorize_performance(score):
    if score >= PERFORMANCE_THRESHOLDS['EXCELLENT']:
        return 'EXCELLENT'
    elif score >= PERFORMANCE_THRESHOLDS['GOOD']:
        return 'GOOD'
    elif score >= PERFORMANCE_THRESHOLDS['AVERAGE']:
        return 'AVERAGE'
    elif score >= PERFORMANCE_THRESHOLDS['NEEDS_IMPROVEMENT']:
        return 'NEEDS_IMPROVEMENT'
    else:
        return 'POOR'

def categorize_risk(risk_score):
    """Categorize risk score"""
    if risk_score >= RISK_THRESHOLDS['HIGH']:
        return 'HIGH', ''
    elif risk_score >= RISK_THRESHOLDS['MEDIUM']:
        return 'MEDIUM', ''
    else:
        return 'LOW', ''

def get_recommendation(performance_score, sentiment_score, risk_score):
    """Get action recommendation based on scores"""
    if performance_score < 65 or sentiment_score < 0.3 or risk_score > 0.5:
        return "IMMEDIATE INTERVENTION REQUIRED"
    elif performance_score < 75 or sentiment_score < 0.5 or risk_score > 0.3:
        return "CLOSE MONITORING NEEDED"
    else:
        return "CONTINUE CURRENT APPROACH"