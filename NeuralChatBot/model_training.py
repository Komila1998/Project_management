from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

from config import LEARNING_RATE
from tensorflow.keras.callbacks import History
import os

def build_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(128, input_shape=(input_dim,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='softmax'))

    sgd = SGD(learning_rate=LEARNING_RATE, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model

def train_model(model, X, Y, epochs, batch_size):
    history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=1)
    return model, history

def save_model(model, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"💾 Model saved to {model_path}")
