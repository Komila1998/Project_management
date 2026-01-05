import os
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import json
from termcolor import colored
from config import MODEL_DIR, STUDENT_FILE, SUPERVISOR_FILE, OUTPUT_FILE

class StudentSupervisorMatcher:
    
    def __init__(self):
        self.load_models()
        
    def load_models(self):
        print(colored("Loading trained models...", "cyan"))
        self.tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
        with open(os.path.join(MODEL_DIR, "domain_classifier.json"), 'r') as f:
            domain_data = json.load(f)
            self.domain_keywords = domain_data['keywords']
        print(colored("Models loaded", "green"))

    def predict_domain(self, topic):
        topic_lower = topic.lower()
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in topic_lower:
                    return domain
        return "General Computing"

    def match_student_to_supervisor(self, student_topic, student_domain, supervisors_df):
        best_match, best_score = None, -1
        for _, supervisor in supervisors_df.iterrows():
            supervisor_text = supervisor['expertise']
            student_text = f"{student_topic} {student_domain}"
            vectors = self.tfidf.transform([student_text, supervisor_text])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            if similarity > best_score:
                best_score = similarity
                best_match = supervisor['name']
                best_expertise = supervisor['expertise']
        return best_match, best_score, best_expertise

    def process_students(self, students_df, supervisors_df):
        results = []
        print(colored("\nProcessing students...", "cyan"))
        for _, student in students_df.iterrows():
            name, topic = student['name'], student['topic']
            domain = self.predict_domain(topic)
            supervisor, score, expertise = self.match_student_to_supervisor(topic, domain, supervisors_df)
            text_features = f"{topic} {domain} {expertise[:50]}..."
            results.append({
                'name': name,
                'topic': topic,
                'domain': domain,
                'text_features': text_features,
                'Assigned_Supervisor': supervisor,
                'similarity_score': score
            })
        return results

    def save_results(self, results):
        df = pd.DataFrame(results)
        df[['name', 'topic', 'domain', 'text_features', 'Assigned_Supervisor']].to_csv(OUTPUT_FILE, index=False)
        print(colored(f"\nResults saved to {OUTPUT_FILE}", "green"))
        return df
