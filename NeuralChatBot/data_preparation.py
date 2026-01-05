import json
import nltk
from nltk.stem import WordNetLemmatizer
from config import DATASET_FILE, IGNORE_WORDS
import pickle
import os
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def load_intents():
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def preprocess_corpus(intents):
    words, classes, documents = [], [], []
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
    
    # Lemmatization
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in IGNORE_WORDS]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    return words, classes, documents

def save_words_classes(words, classes, model_dir):
    import pickle
    os.makedirs(model_dir, exist_ok=True)
    pickle.dump(words, open(os.path.join(model_dir, 'words.pkl'), 'wb'))
    pickle.dump(classes, open(os.path.join(model_dir, 'classes.pkl'), 'wb'))
