import os
import time
import pandas as pd
import numpy as np
import joblib
import openai
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')

#  CONFIGURATIONS -
MODEL_DIR = "models"
STUDENTS_PER_GROUP = 4
CSV_FILE = "new_students.csv"
OUTPUT_FILE = "Result/predicted_groups.csv"

openai.api_key = os.getenv(
    "OPENAI_API_KEY",
    "YOUR_OPEN_API_KEY_HERE"
)



#  PREPROCESSING & MODEL LOADING

def preprocess_new_data(df):
    print(colored("Preprocessing new student data...", "cyan"))

    df = df.copy()
    df.columns = df.columns.str.strip()

    df.fillna({
        "skills": "",
        "interest_area": "",
        "experience_level": "",
        "personality_type": "",
        "domain_interest": ""
    }, inplace=True)

    df["text_features"] = (
        df["skills"] + " " +
        df["interest_area"] + " " +
        df["experience_level"] + " " +
        df["personality_type"] + " " +
        df["domain_interest"]
    )

    print(colored("✓ New data preprocessed", "green"))
    return df


def load_models():
    print(colored("Loading trained ML models...", "cyan"))

    tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    kmeans = joblib.load(os.path.join(MODEL_DIR, "group_model.pkl"))

    print(colored("✓ Models loaded successfully!", "green"))
    return tfidf, scaler, kmeans


def prepare_features(df, tfidf, scaler):
    print(colored("Preparing ML features...", "cyan"))

    text_features = tfidf.transform(df["text_features"])
    numeric = df[["gpa", "availability_hours", "communication_score", "leadership_score"]]
    numeric_scaled = scaler.transform(numeric)

    return np.hstack((numeric_scaled, text_features.toarray()))



#    GROUP BALANCING
def balance_groups(cluster_labels, count):
    balanced = np.full(count, -1)
    sorted_idx = np.argsort(cluster_labels)

    group = 0
    size = 0

    for idx in sorted_idx:
        balanced[idx] = group
        size += 1
        if size >= STUDENTS_PER_GROUP:
            group += 1
            size = 0

    # distribute leftover students
    leftovers = np.where(balanced == -1)[0]
    if len(leftovers) > 0:
        print(colored(f"Distributing {len(leftovers)} leftover students...", "yellow"))

        current_sizes = pd.Series(balanced).value_counts()
        smallest = current_sizes.idxmin()

        for idx in leftovers:
            balanced[idx] = smallest

    return balanced



# TOPIC EVALUATION


def evaluate_topic(topic, description, domain):
    prompt = f"""
    Evaluate this project idea:
    Title: {topic}
    Description: {description}
    Domain: {domain}
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an academic project evaluator."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"Evaluation Error: {str(e)}"



#  PREDICTION PROCESS

def run_prediction():
    print(colored("=" * 60, "cyan"))
    print(colored("GROUP PREDICTION + GPT EVALUATION STARTED", "cyan", attrs=["bold"]))
    print(colored("=" * 60, "cyan"))

    if not os.path.exists(CSV_FILE):
        print(colored(f"Input file not found: {CSV_FILE}", "red"))
        return

    tfidf, scaler, kmeans = load_models()

    df = pd.read_csv(CSV_FILE)
    print(colored(f"Loaded {len(df)} students", "green"))

    df = preprocess_new_data(df)
    features = prepare_features(df, tfidf, scaler)

    print(colored("Predicting groups...", "cyan"))
    df["Cluster"] = kmeans.predict(features)

    print(colored("Balancing groups...", "cyan"))
    df["Group_ID"] = balance_groups(df["Cluster"], len(df))

    evaluations = []
    print(colored("\nEvaluating Topics with GPT...", "magenta"))

    for _, row in df.iterrows():
        print(colored(f"Evaluating: {row['proposed_topic']}", "yellow"))
        e = evaluate_topic(row["proposed_topic"], row["topic_description"], row["domain_interest"])
        evaluations.append(e)
        print(colored("✓ Evaluation Done", "green"))
        time.sleep(1)

    df["LLM_Evaluation"] = evaluations

    # Save output
    os.makedirs("Result", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(colored("\nFINAL RESULT SAVED!", "green", attrs=["bold"]))
    print(colored(f"File: {OUTPUT_FILE}", "yellow"))


if __name__ == "__main__":
    run_prediction()
