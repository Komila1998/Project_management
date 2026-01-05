import pandas as pd
from termcolor import colored
from config import TRAINING_DATA_FILE

def load_training_data():
    print(colored("Loading training dataset...", "cyan"))
    df = pd.read_csv(TRAINING_DATA_FILE)
    print(colored(f"Loaded {len(df)} training records", "green"))
    print(colored("\nTraining Dataset Info:", "cyan"))
    print(df.head())
    return df
