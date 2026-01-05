import os

# Paths -
DATASET_FILE = "ChatbotDataset/dataset.json"
MODEL_DIR = "Trained_NerualModels"
MODEL_FILE = f"{MODEL_DIR}/chatbot.h5"
WORDS_FILE = f"{MODEL_DIR}/words.pkl"
CLASSES_FILE = f"{MODEL_DIR}/classes.pkl"

os.makedirs(MODEL_DIR, exist_ok=True)

#  Training Config
EPOCHS = 100
BATCH_SIZE = 5
LEARNING_RATE = 0.01
CONFIDENCE_THRESHOLD = 0.35
IGNORE_WORDS = ['?', '!']
