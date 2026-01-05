import os
import numpy as np
import pandas as pd
from termcolor import colored
from config import (
    MODEL_FILE, SCALER_FILE, NEW_DATA_FILE, OUTPUT_FILE, TIME_STEPS
)
import text_processor
import metrics_calculator
from data_process import process_new_data

def load_model():
    try:
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_FILE)
        print(colored("Model loaded successfully", "green"))
        return model
    except Exception as e:
        print(colored(f"Error loading model: {e}", "red"))
        return None

def predict_for_new_data():
    print(colored(" Performance Prediction System", "cyan"))
    print(colored("=" * 60, "cyan"))
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Process new data
    X_new, new_df, scaler = process_new_data(NEW_DATA_FILE)
    
    if X_new is None or new_df is None:
        print(colored("No data to predict", "red"))
        return
    
    # Make prediction
    print(colored("\nMaking Predictions...", "cyan"))
    prediction_scaled = model.predict(X_new, verbose=0)
    predicted_score = scaler.inverse_transform(prediction_scaled)[0][0]
    predicted_score = round(predicted_score, 2)
    
    # Display main prediction
    print(colored(f"\nMAIN PREDICTION:", "cyan", attrs=['bold']))
    print(colored(f"   Next Performance Score: {colored(predicted_score, 'green')}/100", "cyan"))
    
    # Analyze latest entry
    display_latest_analysis(new_df, predicted_score)
    
    # Generate predictions for all rows
    results_df = generate_all_predictions(new_df, model, scaler)
    
    # Display summary
    display_prediction_summary(results_df)
    
    # Save results
    save_prediction_results(results_df)
    
    return results_df

def display_latest_analysis(df, predicted_score):
    latest_row = df.iloc[-1]
    
    print(colored("\nLATEST ENTRY ANALYSIS:", "cyan"))
    print(f"   Current Performance: {latest_row['performance_score']}/100")
    print(f"   Predicted Next: {predicted_score}/100")
    
    change = predicted_score - latest_row['performance_score']
    change_str = f"{'+' if change > 0 else ''}{change:.1f}"
    change_color = "green" if change > 0 else "red" if change < 0 else "yellow"
    print(f"   Expected Change: {colored(change_str, change_color)}")
    
    # Sentiment and risk
    sentiment_category = text_processor.categorize_performance(latest_row['sentiment_score'] * 100)
    risk_category, risk_emoji = text_processor.categorize_risk(latest_row['risk_factor'])
    
    print(colored("\nFEEDBACK ANALYSIS:", "cyan"))
    print(f"   Sentiment: {colored(sentiment_category, 'green' if sentiment_category in ['EXCELLENT', 'GOOD'] else 'yellow' if sentiment_category == 'AVERAGE' else 'red')}")
    print(f"   Feedback Preview: \"{latest_row['feedback_review'][:60]}...\"")
    
    print(colored("\nRISK ASSESSMENT:", "cyan"))
    print(f"   Risk Level: {risk_emoji} {colored(risk_category, 'red' if risk_category == 'HIGH' else 'yellow' if risk_category == 'MEDIUM' else 'green')}")
    print(f"   Recommended Actions: {latest_row['recommended_actions'][:60]}...")

def generate_all_predictions(df, model, scaler):
    print(colored("\nGENERATING PREDICTIONS FOR ALL ROWS...", "cyan"))
    
    predictions = []
    confidences = []
    
    performance_data = df['performance_score'].values.reshape(-1, 1)
    scaled_data = scaler.transform(performance_data)
    
    for i in range(len(scaled_data)):
        # Calculate confidence based on available history
        history_length = min(i, TIME_STEPS)
        confidence = metrics_calculator.calculate_prediction_confidence(history_length, TIME_STEPS)
        
        if i >= TIME_STEPS:
            # Full history available
            X_seq = scaled_data[i-TIME_STEPS:i].reshape(1, TIME_STEPS, 1)
            pred_scaled = model.predict(X_seq, verbose=0)
            pred_score = scaler.inverse_transform(pred_scaled)[0][0]
        elif i > 0:
            # Partial history
            X_seq = scaled_data[:i].reshape(1, i, 1)
            if i < TIME_STEPS:
                padding = np.zeros((TIME_STEPS - i, 1))
                X_seq = np.concatenate([padding, scaled_data[:i]]).reshape(1, TIME_STEPS, 1)
            pred_scaled = model.predict(X_seq, verbose=0)
            pred_score = scaler.inverse_transform(pred_scaled)[0][0]
        else:
            # No history
            pred_score = performance_data[i][0]
        
        predictions.append(round(pred_score, 2))
        confidences.append(round(confidence, 2))
    
    # Create results dataframe
    results_df = df.copy()
    results_df['predicted_next_score'] = predictions
    results_df['prediction_confidence'] = confidences
    
    # Calculate additional metrics
    results_df['previous_score'] = results_df['performance_score'].shift(1)
    results_df['trend'] = results_df.apply(
        lambda row: metrics_calculator.calculate_trend(
            row['predicted_next_score'], row['previous_score']
        ), axis=1
    )
    
    results_df['risk_status'] = results_df.apply(
        lambda row: metrics_calculator.calculate_risk_status(
            row['predicted_next_score'], row['sentiment_score'], row['risk_factor']
        ), axis=1
    )
    
    results_df['recommended_action'] = results_df.apply(
        lambda row: text_processor.get_recommendation(
            row['predicted_next_score'], row['sentiment_score'], row['risk_factor']
        ), axis=1
    )
    
    return results_df

def display_prediction_summary(results_df):

    print(colored("\nPREDICTION SUMMARY", "cyan"))
    print(colored("-" * 40, "cyan"))
    
    # Trend analysis
    trend_counts = results_df['trend'].value_counts()
    print(colored("\nTREND ANALYSIS:", "cyan"))
    for trend, count in trend_counts.items():
        percentage = (count / len(results_df)) * 100
        print(f"   {trend}: {count} rows ({percentage:.1f}%)")
    
    # Risk distribution
    risk_counts = results_df['risk_status'].value_counts()
    print(colored("\nRISK DISTRIBUTION:", "cyan"))
    for risk, count in risk_counts.items():
        percentage = (count / len(results_df)) * 100
        risk_emoji = "" if risk == "HIGH RISK" else "🟡" if risk == "MEDIUM RISK" else "🟢"
        print(f"   {risk_emoji} {risk}: {count} rows ({percentage:.1f}%)")
    
    # Action recommendations
    action_counts = results_df['recommended_action'].value_counts()
    print(colored("\n ACTION RECOMMENDATIONS:", "cyan"))
    for action, count in action_counts.items():
        print(f"   {action}: {count}")
    
    # Performance statistics
    avg_current = results_df['performance_score'].mean()
    avg_predicted = results_df['predicted_next_score'].mean()
    avg_confidence = results_df['prediction_confidence'].mean()
    
    print(colored("\n PERFORMANCE STATISTICS:", "cyan"))
    print(f"   Average Current Score: {avg_current:.1f}")
    print(f"   Average Predicted Score: {avg_predicted:.1f}")
    print(f"   Average Prediction Confidence: {avg_confidence:.1%}")
    print(f"   Overall Trend: {colored('IMPROVING' if avg_predicted > avg_current else 'DECLINING' if avg_predicted < avg_current else 'STABLE', 'green' if avg_predicted > avg_current else 'red' if avg_predicted < avg_current else 'yellow')}")

def save_prediction_results(results_df):

    # Save detailed results
    results_df.to_csv(OUTPUT_FILE, index=False)
    
 
    
    print(colored(f"\n Results saved to:", "green"))
    print(colored(f"   Detailed predictions: {OUTPUT_FILE}", "cyan"))
 

if __name__ == "__main__":
    predict_for_new_data()