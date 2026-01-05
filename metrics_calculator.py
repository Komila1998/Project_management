# metrics_calculator.py
import pandas as pd
import numpy as np
from config import PERFORMANCE_THRESHOLDS

def calculate_trend(current_score, previous_score):
    if pd.isna(previous_score):
        return "FIRST"
    
    change = current_score - previous_score
    if change > 2:
        return "IMPROVING"
    elif change < -2:
        return "DECLINING"
    else:
        return "STABLE"

def calculate_risk_status(performance_score, sentiment_score, risk_score):
    """Calculate overall risk status"""
    if performance_score < 65 or sentiment_score < 0.3 or risk_score > 0.5:
        return "HIGH RISK"
    elif performance_score < 75 or sentiment_score < 0.5 or risk_score > 0.3:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"

def calculate_prediction_confidence(history_length, time_steps=3):
    if history_length >= time_steps:
        return 0.9  # High confidence with full history
    elif history_length > 0:
        return 0.3 + (0.6 * (history_length / time_steps))
    else:
        return 0.1  # Low confidence with no history

def get_performance_summary(df):
    summary = {
        'total_rows': len(df),
        'avg_performance': round(df['performance_score'].mean(), 2),
        'min_performance': df['performance_score'].min(),
        'max_performance': df['performance_score'].max(),
        'std_performance': round(df['performance_score'].std(), 2),
        'avg_sentiment': round(df['sentiment_score'].mean(), 2),
        'avg_risk': round(df['risk_factor'].mean(), 2)
    }
    
    return summary

def get_prediction_accuracy(actual_scores, predicted_scores):
    if len(actual_scores) != len(predicted_scores):
        return None
    
    errors = np.abs(np.array(actual_scores) - np.array(predicted_scores))
    
    metrics = {
        'mae': round(np.mean(errors), 2),  # Mean Absolute Error
        'rmse': round(np.sqrt(np.mean(errors**2)), 2),  # Root Mean Square Error
        'max_error': round(np.max(errors), 2),
        'accuracy_within_5': round((errors <= 5).sum() / len(errors) * 100, 1),
        'accuracy_within_10': round((errors <= 10).sum() / len(errors) * 100, 1)
    }
    
    return metrics