from data_preparation import load_training_data
from feature_engineering import train_tfidf_model
from model_training import create_domain_classifier

def main():
    df = load_training_data()
    train_tfidf_model(df)
    create_domain_classifier()
    print("Training Completed!")

if __name__ == "__main__":
    main()
