import pandas as pd
import os
from termcolor import colored

CSV_FILE = "Dataset/Dataset.csv"
PROCESSED_FILE = "processed_data/processed_students.csv"

def load_and_prepare_data():
    print(colored("Loading raw dataset...", "cyan"))
    df = pd.read_csv(CSV_FILE)
    df.columns = df.columns.str.strip()
    print(colored("Dataset loaded successfully!\n", "green"))
    print(df.head(), "\n")

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

    df.to_csv(PROCESSED_FILE, index=False)
    print(colored(f"Cleaned data saved to {PROCESSED_FILE}\n", "green"))
    return df

if __name__ == "__main__":
    load_and_prepare_data()
