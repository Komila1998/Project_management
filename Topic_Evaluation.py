
import os
import time
import pandas as pd
import numpy as np
import joblib
import openai
from termcolor import colored
from feature_engineering import FEATURE_FILE        
from data_preparation import PROCESSED_FILE        
from model_training import MODEL_DIR               


openai.api_key = os.getenv(
    "OPENAI_API_KEY",
    "YOUR_OPEN_API_KEY"  # Replace or use env variable


)

print(colored("Loading trained models...", "cyan"))
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
kmeans = joblib.load(os.path.join(MODEL_DIR, "group_model.pkl"))
print(colored("Models loaded successfully!\n", "green"))

CSV_FILE = "new_students.csv"
 
df_new = pd.read_csv(CSV_FILE)
df_new.columns = df_new.columns.str.strip()

df_new["text_features"] = (
    df_new["skills"].fillna('') + " " +
    df_new["interest_area"].fillna('') + " " +
    df_new["experience_level"].fillna('') + " " +
    df_new["personality_type"].fillna('') + " " +
    df_new["domain_interest"].fillna('')
)

text_features = tfidf.transform(df_new["text_features"])
numeric = df_new[["gpa", "availability_hours", "communication_score", "leadership_score"]]
numeric_scaled = scaler.transform(numeric)
combined_features = np.hstack((numeric_scaled, text_features.toarray()))

print(colored("Predicting group assignments...", "cyan"))
df_new["Predicted_Group"] = kmeans.predict(combined_features)
print(colored("Prediction complete!\n", "green"))

def evaluate_topic(topic, description, domain):
    prompt = f"""
    Evaluate this project idea:
    Title: {topic}
    Description: {description}
    Domain: {domain}
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an academic project evaluator."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message["content"]

evaluations = []
for _, row in df_new.iterrows():
    print(colored(f"Evaluating: {row['proposed_topic']}", "yellow"))
    result = evaluate_topic(row["proposed_topic"], row["topic_description"], row["domain_interest"])
    print(colored(result, "green"))
    evaluations.append(result)
    time.sleep(1)

df_new["LLM_Evaluation"] = evaluations
df_new.to_csv("Result/predicted_groups.csv", index=False)

print(colored("Saved predictions to predicted_groups.csv", "green"))
print(colored("Prediction + Topic Evaluation completed!", "magenta"))
