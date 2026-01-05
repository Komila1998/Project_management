import os

# Paths
TRAINING_DATA_FILE = "data/training_data.csv"
STUDENT_FILE = "InputData/students.csv"
SUPERVISOR_FILE = "InputData/supervisors.csv"
MODEL_DIR = "models"
OUTPUT_FILE = "output/result.csv"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

#  Domain Keywords
DOMAIN_KEYWORDS = {
    'Artificial Intelligence': ['ai', 'artificial intelligence', 'machine learning', 'deep learning', 'neural network', 'computer vision'],
    'Blockchain': ['blockchain', 'cryptography', 'distributed ledger', 'smart contract', 'voting system'],
    'Internet of Things': ['iot', 'internet of things', 'smart irrigation', 'edge computing', 'embedded systems'],
    'Cybersecurity': ['cybersecurity', 'network security', 'privacy', 'security', 'encryption'],
    'Natural Language Processing': ['nlp', 'natural language processing', 'sentiment analysis', 'text mining', 'linguistics', 'sinhala', 'tweets']
}
