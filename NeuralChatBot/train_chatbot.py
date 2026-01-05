from data_preparation import load_intents, preprocess_corpus, save_words_classes
from feature_engineering import create_training_data
from model_training import build_model, train_model, save_model
from config import MODEL_DIR, MODEL_FILE, EPOCHS, BATCH_SIZE

def main():
    intents = load_intents()
    words, classes, documents = preprocess_corpus(intents)
    save_words_classes(words, classes, MODEL_DIR)
    X, Y = create_training_data(documents, words, classes)
    
    model = build_model(input_dim=X.shape[1], output_dim=Y.shape[1])
    model, history = train_model(model, X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    save_model(model, MODEL_FILE)
    
    print("\nChatbot Training Completed!")

if __name__ == "__main__":
    main()
