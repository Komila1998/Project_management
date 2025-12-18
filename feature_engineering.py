import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from termcolor import colored
import os
from data_preparation import PROCESSED_FILE

FEATURE_FILE = "student_features.npy"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def generate_features():
    print(colored("Loading processed dataset...", "cyan"))
    df = pd.read_csv(PROCESSED_FILE)

    print(colored("⚙ Performing TF-IDF vectorization...", "cyan"))
    tfidf = TfidfVectorizer(stop_words="english")
    text_features = tfidf.fit_transform(df["text_features"])

    numeric_features = df[["gpa", "availability_hours", "communication_score", "leadership_score"]]
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_features)

    combined_features = np.hstack((numeric_scaled, text_features.toarray()))
    np.save(FEATURE_FILE, combined_features)

    joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    print(colored(f"Features saved as {FEATURE_FILE} and vectorizers stored.\n", "green"))
    return combined_features


if __name__ == "__main__":
    generate_features()
