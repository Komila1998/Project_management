import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from config import TIME_STEPS, SCALER_FILE
import text_processor

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def prepare_training_data(self, data_path):
        print("Loading training data...")
        
        df = pd.read_csv(data_path)
        print(f"Training data shape: {df.shape}")
        
        # Extract sentiment and risk scores
        df = self._extract_metrics(df)
        
        # Prepare performance sequences
        X, y = self._create_sequences(df['performance_score'].values)
        
        # Fit scaler and transform
        self.scaler.fit(df['performance_score'].values.reshape(-1, 1))
        X_scaled = self.scaler.transform(X.reshape(-1, 1)).reshape(X.shape)
        y_scaled = self.scaler.transform(y.reshape(-1, 1)).reshape(y.shape)
        
        # Save scaler
        self._save_scaler()
        
        return X_scaled, y_scaled, df
    
    def prepare_prediction_data(self, data_path, use_existing_scaler=True):
        print(f"Loading prediction data from {data_path}...")
        
        try:
            df = pd.read_csv(data_path)
            print(f"Prediction data shape: {df.shape}")
            
            # Extract sentiment and risk scores
            df = self._extract_metrics(df)
            
            # Load or use existing scaler
            if use_existing_scaler:
                self.scaler = joblib.load(SCALER_FILE)
            
            # Prepare input sequence
            X_input = self._prepare_input_sequence(df['performance_score'].values)
            
            return X_input, df
            
        except FileNotFoundError:
            print(f"File '{data_path}' not found!")
            return None, None
    
    def _extract_metrics(self, df):
        if 'sentiment_score' not in df.columns:
            df['sentiment_score'] = df['feedback_review'].apply(
                text_processor.extract_sentiment_score
            )
        
        if 'risk_factor' not in df.columns:
            df['risk_factor'] = df['recommended_actions'].apply(
                text_processor.calculate_risk_factor
            )
        
        return df
    
    def _create_sequences(self, data):
        X, y = [], []
        
        for i in range(len(data) - TIME_STEPS):
            X.append(data[i:i + TIME_STEPS])
            y.append(data[i + TIME_STEPS])
        
        X = np.array(X).reshape(-1, TIME_STEPS, 1)
        y = np.array(y).reshape(-1, 1)
        
        print(f"Created sequences: X={X.shape}, y={y.shape}")
        
        return X, y
    
    def _prepare_input_sequence(self, data):
        data_scaled = self.scaler.transform(data.reshape(-1, 1))
        
        if len(data_scaled) >= TIME_STEPS:
            X_input = data_scaled[-TIME_STEPS:].reshape(1, TIME_STEPS, 1)
            print(f"   Using last {TIME_STEPS} scores")
        else:
            # Pad with zeros if not enough data
            padding = np.zeros((TIME_STEPS - len(data_scaled), 1))
            X_input = np.vstack([padding, data_scaled]).reshape(1, TIME_STEPS, 1)
            print(f"   Using {len(data_scaled)} scores (padded)")
        
        return X_input
    
    def _save_scaler(self):

        from config import MODEL_DIR
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(self.scaler, SCALER_FILE)
        print(f"Scaler saved to {SCALER_FILE}")