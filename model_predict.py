import os
import pandas as pd
import numpy as np
import joblib
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR = "models"
STUDENTS_PER_GROUP = 4

def preprocess_new_data(new_df):
    print(colored("Preprocessing new student data...", "cyan"))
    
    df = new_df.copy()
    
    # Strip column names if needed
    df.columns = df.columns.str.strip()
    
    # Handle missing values
    df.fillna({
        "skills": "",
        "interest_area": "",
        "experience_level": "",
        "personality_type": "",
        "domain_interest": ""
    }, inplace=True)
    
    # Create text_features column (same as training)
    df["text_features"] = (
        df["skills"] + " " +
        df["interest_area"] + " " +
        df["experience_level"] + " " +
        df["personality_type"] + " " +
        df["domain_interest"]
    )
    
    print(colored("✓ New data preprocessed successfully", "green"))
    return df

def load_trained_model():

    print(colored("Loading trained model...", "cyan"))
    
    # Check if model files exist
    model_path = os.path.join(MODEL_DIR, "group_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    tfidf_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    
    if missing:
    print("WARNING: Missing files:")
    for p in missing:
        print(" -", p)
    print("Training not found. Skipping load...")
    # return / exit / or trigger training here
    
    # Load model, scaler, and TF-IDF vectorizer
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    tfidf_vectorizer = joblib.load(tfidf_path)
    
    print(colored("Model, scaler, and TF-IDF vectorizer loaded successfully", "green"))
    return model, scaler, tfidf_vectorizer

def prepare_features_for_prediction(df, tfidf_vectorizer, scaler):

    print(colored("Preparing features for prediction...", "cyan"))
    
    # Get numeric features
    numeric_cols = ["gpa", "availability_hours", "communication_score", "leadership_score"]
    
    # Check if all numeric columns exist
    missing_numeric = [col for col in numeric_cols if col not in df.columns]
    if missing_numeric:
        raise ValueError(f"Missing numeric columns: {missing_numeric}")
    
    numeric_features = df[numeric_cols].values
    
    # Get text features using saved TF-IDF vectorizer
    if "text_features" not in df.columns:
        raise ValueError("Column 'text_features' not found. Make sure preprocessing created it.")
    
    text_features = tfidf_vectorizer.transform(df["text_features"])
    
    # Combine features
    combined_features = np.hstack((numeric_features, text_features.toarray()))
    
    print(f"  Numeric features shape: {numeric_features.shape}")
    print(f"  Text features shape: {text_features.shape}")
    print(f"  Combined features shape: {combined_features.shape}")
    
    # 4. Scale numeric features using saved scaler
    combined_features_scaled = combined_features.copy()
    combined_features_scaled[:, :4] = scaler.transform(combined_features[:, :4])
    
    return combined_features_scaled

def predict_groups_for_new_students(new_students_csv, output_csv="predicted_groups.csv"):

    print(colored("=" * 60, "cyan"))
    print(colored("PREDICTION PHASE", "cyan", attrs=["bold"]))
    print(colored("=" * 60, "cyan"))
    
    # Load the trained model and transformers
    model, scaler, tfidf_vectorizer = load_trained_model()
    
    # Load new student data
    print(colored("\nSTEP 1: Loading new student data...", "cyan"))
    if not os.path.exists(new_students_csv):
        raise FileNotFoundError(f"Input file not found: {new_students_csv}")
    
    new_df = pd.read_csv(new_students_csv)
    print(f"  Loaded {len(new_df)} new students")
    print(f"  Columns: {list(new_df.columns)}")
    
    # Preprocess the data
    print(colored("\nSTEP 2: Preprocessing data...", "cyan"))
    processed_df = preprocess_new_data(new_df)
    
    # Prepare features using saved transformers
    print(colored("\nSTEP 3: Preparing features...", "cyan"))
    X_new = prepare_features_for_prediction(processed_df, tfidf_vectorizer, scaler)
    
    # Check feature dimensions
    expected_features = 4 + tfidf_vectorizer.transform([""]).shape[1]
    print(f"  Expected features: {expected_features}")
    print(f"  Actual features: {X_new.shape[1]}")
    
    if X_new.shape[1] != expected_features:
        print(colored(f"Warning: Feature dimension mismatch! Expected {expected_features}, got {X_new.shape[1]}", "yellow"))
        print("  Trying to handle dimension mismatch...")
        
        # Handle dimension mismatch by padding or truncating
        if X_new.shape[1] > expected_features:
            X_new = X_new[:, :expected_features]
            print(f"  Truncated to {expected_features} features")
        else:
            # Pad with zeros
            padding = np.zeros((X_new.shape[0], expected_features - X_new.shape[1]))
            X_new = np.hstack([X_new, padding])
            print(f"  Padded to {expected_features} features")
    
    #  Predict clusters using the trained model
    print(colored("\nSTEP 4: Predicting clusters...", "cyan"))
    cluster_labels = model.predict(X_new)
    
    # Balance groups to ensure 4 students per group
    print(colored("\nSTEP 5: Balancing groups...", "cyan"))
    balanced_groups = balance_new_groups(cluster_labels, len(new_df))
    
    # Assign groups to students
    processed_df["Predicted_Cluster"] = cluster_labels
    processed_df["Group_ID"] = balanced_groups
    processed_df["Group_Size"] = STUDENTS_PER_GROUP
    
    # Add original columns back
    for col in new_df.columns:
        if col not in processed_df.columns:
            processed_df[col] = new_df[col]
    
    # Save predictions
    print(colored("\nSTEP 6: Saving predictions...", "cyan"))
    processed_df.to_csv(output_csv, index=False)

    print_results(processed_df, output_csv)
    
    return processed_df

def balance_new_groups(cluster_labels, num_students):

    target_groups = num_students // STUDENTS_PER_GROUP
    remainder = num_students % STUDENTS_PER_GROUP
    
    if remainder > 0:
        print(f"  Note: {remainder} students will be distributed to existing groups")
    
    # Create balanced groups
    balanced_labels = np.full(num_students, -1, dtype=int)
    
    # Sort by cluster for better grouping
    sorted_indices = np.argsort(cluster_labels)
    group_id = 0
    current_size = 0
    
    for idx in sorted_indices:
        balanced_labels[idx] = group_id
        current_size += 1
        
        if current_size >= STUDENTS_PER_GROUP:
            group_id += 1
            current_size = 0
    
    # Handle any remaining students
    unassigned = np.where(balanced_labels == -1)[0]
    if len(unassigned) > 0:
        print(f"  Distributing {len(unassigned)} remaining students...")
        # Assign remaining students to smallest groups
        for idx in unassigned:
            group_sizes = pd.Series(balanced_labels[balanced_labels != -1]).value_counts()
            smallest_group = group_sizes.idxmin()
            balanced_labels[idx] = smallest_group
    
    return balanced_labels

def print_results(predicted_df, output_file):

    print(colored("\n" + "=" * 60, "green"))
    print(colored("PREDICTION COMPLETED!", "green", attrs=["bold"]))
    print(colored("=" * 60, "green"))
    
    print(colored("\nPrediction Summary:", "yellow"))
    print(f"  Total students predicted: {len(predicted_df)}")
    print(f"  Number of groups formed: {predicted_df['Group_ID'].nunique()}")
    print(f"  Target group size: {STUDENTS_PER_GROUP}")
    
    # Check group sizes
    group_sizes = predicted_df["Group_ID"].value_counts()
    print("\nGroup sizes:")
    for group_id, size in group_sizes.items():
        print(f"  Group {group_id}: {size} students")
    
    balanced = all(size == STUDENTS_PER_GROUP for size in group_sizes)
    
    if balanced:
        print(colored("\n✓ All groups have exactly 4 students!", "green"))
    else:
        print(colored(f"\nSome groups don't have 4 students", "yellow"))
    
    # Display group details
    print(colored("\nGroup Details (first 5 groups):", "cyan"))
    for group_id in sorted(predicted_df["Group_ID"].unique())[:5]:
        group_members = predicted_df[predicted_df["Group_ID"] == group_id]
        print(f"\nGroup {group_id} ({len(group_members)} students):")
        
        # Calculate group statistics
        if 'gpa' in group_members.columns:
            avg_gpa = group_members["gpa"].mean()
            print(f"  Average GPA: {avg_gpa:.2f}")
        
        if 'communication_score' in group_members.columns:
            avg_comm = group_members["communication_score"].mean()
            print(f"  Average Communication: {avg_comm:.1f}")
        
        if 'leadership_score' in group_members.columns:
            avg_lead = group_members["leadership_score"].mean()
            print(f"  Average Leadership: {avg_lead:.1f}")
        
        print("  Members:")
        for idx, row in group_members.iterrows():
            skills_preview = row['skills'][:30] + "..." if len(row['skills']) > 30 else row['skills']
            print(f"    - {skills_preview}")
    
    print(colored(f"\nPredictions saved to: {output_file}", "yellow"))

if __name__ == "__main__":
    #  input and output file paths
    input_file = "new_students.csv"
    output_file = "Result/predicted_groups.csv"
    
    print(colored("=" * 60, "cyan"))
    print(colored("GROUP PREDICTION STARTED", "cyan", attrs=["bold"]))
    print(colored("=" * 60, "cyan"))
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(colored(f"\nError: Input file not found at {input_file}", "red"))
            print("\nPlease:")
            print(f"1. Create a file named '{input_file}' in the current directory")
            print("2. Ensure it has these exact columns:")
            print("   gpa, skills, interest_area, experience_level, personality_type,")
            print("   availability_hours, communication_score, leadership_score,")
            print("   proposed_topic, topic_description, domain_interest")
            

            exit(1)
        
        # Run prediction
        predicted_df = predict_groups_for_new_students(input_file, output_file)
        print(colored(f"\nPredictions successfully saved to {output_file}", "green", attrs=["bold"]))
        
    except FileNotFoundError as e:
        print(colored(f"\nFile Error: {e}", "red"))
        print("\nTo train the model first, run:")
        print(colored("python model_training.py", "yellow"))
        
    except Exception as e:
        print(colored(f"\nError: {e}", "red"))
        import traceback
        print("\nDetailed error traceback:")
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Run model_training.py first to generate all model files")
        print("2. Make sure models/ folder contains:")
        print("   - group_model.pkl")
        print("   - scaler.pkl")
        print("   - tfidf_vectorizer.pkl")
        print("3. Check new_students.csv has all required columns")

