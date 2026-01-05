import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import IGNORE_WORDS
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def create_training_data(documents, words, classes):
    training = []
    output_empty = [0] * len(classes)
    
    for doc in documents:
        bag = []
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
        for w in words:
            bag.append(1 if w in pattern_words else 0)
        output_row = output_empty.copy()
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])
    
    np.random.shuffle(training)
    
    X = np.array([sample[0] for sample in training])
    Y = np.array([sample[1] for sample in training])
    
    # Pad sequences
    X = pad_sequences(X, maxlen=len(words), padding='post')
    
    return X, Y
