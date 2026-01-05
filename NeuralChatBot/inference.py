import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import MODEL_FILE, WORDS_FILE, CLASSES_FILE, DATASET_FILE, CONFIDENCE_THRESHOLD

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

# Load model and artifacts
model = load_model(MODEL_FILE)
words = pickle.load(open(WORDS_FILE, 'rb'))
classes = pickle.load(open(CLASSES_FILE, 'rb'))
intents = json.load(open(DATASET_FILE, 'r', encoding='utf-8'))

def clean_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in tokens]

def bow(sentence, words_list):
    sentence_words = clean_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words_list]
    bag = pad_sequences([bag], maxlen=len(words_list), padding='post')[0]
    return np.array(bag)

def predict_intent(text):
    bow_data = bow(text, words)
    res = model.predict(np.array([bow_data]), verbose=0)[0]
    predictions = [(classes[i], float(res[i])) for i in range(len(res))]
    predictions.sort(key=lambda x: x[1], reverse=True)
    if predictions[0][1] < CONFIDENCE_THRESHOLD:
        return "no_match", predictions[0][1]
    return predictions[0][0], predictions[0][1]

def get_response_from_intent(intent_tag):
    for intent in intents['intents']:
        if intent['tag'] == intent_tag:
            return np.random.choice(intent['responses'])
    return "I'm not sure I understand. Can you explain differently?"

def chatbot_response(user_text):
    intent_tag, confidence = predict_intent(user_text)
    if intent_tag == "no_match":
        return {"intent": "no_match", "confidence": confidence,
                "response": "I'm not sure I understand. Please say it differently."}
    response = get_response_from_intent(intent_tag)
    return {"intent": intent_tag, "confidence": confidence, "response": response}
