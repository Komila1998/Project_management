from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from termcolor import colored
from config import MODEL_DIR

def train_tfidf_model(df):
    print(colored("\nTraining TF-IDF Vectorizer...", "cyan"))
    training_text = df['topic'] + " " + df['domain'] + " " + df['expertise']
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1,2))
    tfidf.fit(training_text)
    
    vectorizer_path = f"{MODEL_DIR}/tfidf_vectorizer.pkl"
    joblib.dump(tfidf, vectorizer_path)
    print(colored(f"Vectorizer saved to {vectorizer_path}", "green"))
    return tfidf
