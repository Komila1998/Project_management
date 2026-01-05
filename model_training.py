import json
from termcolor import colored
from config import MODEL_DIR, DOMAIN_KEYWORDS

def create_domain_classifier():
    print(colored("\n🏷Creating domain classifier...", "cyan"))
    domain_data = {'keywords': DOMAIN_KEYWORDS, 'domains': list(DOMAIN_KEYWORDS.keys())}
    domain_path = f"{MODEL_DIR}/domain_classifier.json"
    
    with open(domain_path, 'w') as f:
        json.dump(domain_data, f, indent=2)
    
    print(colored(f"Domain classifier saved to {domain_path}", "green"))
    return DOMAIN_KEYWORDS

def predict_domain(topic, domain_keywords):
    topic_lower = topic.lower()
    for domain, keywords in domain_keywords.items():
        for keyword in keywords:
            if keyword in topic_lower:
                return domain
    return "General Computing"
