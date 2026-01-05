import pandas as pd
import numpy as np
from termcolor import colored
import joblib
from config import TRAINING_DATA_FILE, setup_directories
import text_processor
from data_preprocessor import DataPreprocessor
import metrics_calculator

def process_training_data():

    print(colored("📥 Processing training data...", "cyan"))
    
    # Setup directories
    setup_directories()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Prepare training data
    X, y, df = preprocessor.prepare_training_data(TRAINING_DATA_FILE)
    
    print(colored(f"Training data processed successfully!", "green"))
    print(colored(f"Training sequences: X={X.shape}, y={y.shape}", "cyan"))
    
    # Calculate and display metrics
    metrics = metrics_calculator.get_performance_summary(df)
    print(colored("\nTraining Data Summary:", "cyan"))
    for key, value in metrics.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    return X, y, df, preprocessor.scaler

def process_new_data(new_data_file="newData.csv"):
    print(colored(f"Processing new data from {new_data_file}...", "cyan"))
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Prepare prediction data
    X_new, new_df = preprocessor.prepare_prediction_data(new_data_file)
    
    if X_new is None or new_df is None:
        return None, None, None
    
    print(colored(f"New data processed successfully!", "green"))
    print(colored(f"New data shape: {new_df.shape}", "cyan"))
    
    # Display new data metrics
    metrics = metrics_calculator.get_performance_summary(new_df)
    print(colored("\nNew Data Summary:", "cyan"))
    print(f"   Rows: {metrics['total_rows']}")
    print(f"   Avg Performance: {metrics['avg_performance']}")
    print(f"   Avg Sentiment: {metrics['avg_sentiment']}")
    print(f"   Avg Risk: {metrics['avg_risk']}")
    
    return X_new, new_df, preprocessor.scaler

if __name__ == "__main__":
    process_training_data()