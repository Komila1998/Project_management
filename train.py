import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from termcolor import colored

# Local imports
from config import (
    MODEL_DIR, MODEL_FILE, TIME_STEPS, LSTM_UNITS, 
    DENSE_UNITS, DROPOUT_RATE, BATCH_SIZE, EPOCHS, VALIDATION_SPLIT
)
from data_process import process_training_data

def build_model(input_shape):

    model = Sequential([
        LSTM(LSTM_UNITS, return_sequences=True, input_shape=input_shape),
        Dropout(DROPOUT_RATE),
        LSTM(DENSE_UNITS),
        Dropout(DROPOUT_RATE),
        Dense(1)  # Predict single performance score
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model():
    print(colored("Training LSTM Model for Performance Prediction", "cyan"))
    print("=" * 60)
    
    # Get training data
    X, y, df, scaler = process_training_data()
    
    if X is None or y is None:
        print(colored("Failed to load training data", "red"))
        return
    
    print(colored("\nBuilding Model Architecture...", "cyan"))
    print(colored(f"   Input Shape: {X.shape[1:]}", "cyan"))
    print(colored(f"   LSTM Units: {LSTM_UNITS}", "cyan"))
    print(colored(f"   Dense Units: {DENSE_UNITS}", "cyan"))
    print(colored(f"   Dropout Rate: {DROPOUT_RATE}", "cyan"))
    
    # Build model
    model = build_model(X.shape[1:])
    
    print(colored("\nModel Summary:", "cyan"))
    model.summary()
    
    print(colored("\nTraining Model...", "cyan"))
    history = model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=1
    )
    
    # Save model
    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_FILE)
    print(colored(f"\nModel saved to '{MODEL_FILE}'", "green"))
    
    # Training results
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_mae = history.history['mae'][-1]
    final_val_mae = history.history['val_mae'][-1]
    
    print(colored("\nTraining Results:", "cyan"))
    print(f"   Final Training Loss: {final_loss:.6f}")
    print(f"   Final Validation Loss: {final_val_loss:.6f}")
    print(f"   Final Training MAE: {final_mae:.4f}")
    print(f"   Final Validation MAE: {final_val_mae:.4f}")
    print(f"   Training Samples: {len(X)}")
    print(f"   Epochs Trained: {EPOCHS}")
    
    return model

if __name__ == "__main__":
    train_model()