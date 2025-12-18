import os
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from termcolor import colored

# Local pipeline modules
from data_preparation import load_and_prepare_data, PROCESSED_FILE
from feature_engineering import generate_features, FEATURE_FILE

MODEL_DIR = "models"
GROUPED_FILE = "processed_data/grouped_students.csv"
STUDENTS_PER_GROUP = 4

os.makedirs(MODEL_DIR, exist_ok=True)

def balance_groups_to_four(df, features):

    print(colored("STEP 4: Balancing groups to 4 students each...", "cyan"))
    
    # Calculate number of groups needed
    num_students = len(df)
    target_groups = num_students // STUDENTS_PER_GROUP
    remainder = num_students % STUDENTS_PER_GROUP
    
    print(f"  Total students: {num_students}")
    print(f"  Target groups: {target_groups}")
    print(f"  Students per group: {STUDENTS_PER_GROUP}")
    
    if remainder > 0:
        print(colored(f"  Warning: {remainder} students will be distributed to existing groups", "yellow"))
    
    # Strategy 1: Use KMeans with target groups
    kmeans = KMeans(n_clusters=target_groups, random_state=42, n_init=10)
    initial_labels = kmeans.fit_predict(features)
    
    # Create balanced groups
    balanced_labels = np.full(len(df), -1, dtype=int)
    group_id = 0
    current_group_size = 0
    
    # Sort students by their cluster for better grouping
    sorted_indices = np.argsort(initial_labels)
    
    for idx in sorted_indices:
        balanced_labels[idx] = group_id
        current_group_size += 1
        
        # If group is full, move to next group
        if current_group_size >= STUDENTS_PER_GROUP:
            group_id += 1
            current_group_size = 0
    
    # Handle any remaining students (distribute them to existing groups)
    unassigned_mask = balanced_labels == -1
    if unassigned_mask.any():
        unassigned_indices = np.where(unassigned_mask)[0]
        print(f"  Distributing {len(unassigned_indices)} remaining students...")
        
        # Distribute remaining students to groups with smallest sizes
        for idx in unassigned_indices:
            # Find group with smallest current size
            group_sizes = pd.Series(balanced_labels[balanced_labels != -1]).value_counts()
            smallest_group = group_sizes.idxmin()
            balanced_labels[idx] = smallest_group
    
    # Verify all groups are balanced
    final_groups = pd.Series(balanced_labels)
    group_sizes = final_groups.value_counts().sort_index()
    
    print("\n  Group sizes after balancing:")
    for group_id, size in group_sizes.items():
        print(f"    Group {group_id}: {size} students")
    
    return balanced_labels, kmeans


def train_group_model():

    print(colored("=" * 60, "cyan"))
    print(colored("MODEL TRAINING PHASE", "cyan", attrs=["bold"]))
    print(colored("=" * 60, "cyan"))
    
    print(colored("\nSTEP 1: Preparing data...", "cyan"))
    load_and_prepare_data()

    print(colored("STEP 2: Generating features...", "cyan"))
    generate_features()

    print(colored("STEP 3: Loading feature data...", "cyan"))
    X = np.load(FEATURE_FILE)
    df = pd.read_csv(PROCESSED_FILE)
    
    # Scale features for better clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler for later use in prediction
    joblib.dump(scaler, os.path.join(MODEL_DIR, "feature_scaler.pkl"))

    # Train model with balanced groups
    balanced_labels, kmeans = balance_groups_to_four(df, X_scaled)
    df["Group_ID"] = balanced_labels
    
    # Add Group_Size column
    df["Group_Size"] = STUDENTS_PER_GROUP
    
    # Calculate group statistics
    group_stats = df.groupby("Group_ID").agg({
        "gpa": ["mean", "std"],
        "communication_score": "mean",
        "leadership_score": "mean",
        "availability_hours": "mean"
    }).round(2)
    
    print(colored("\nSTEP 5: Saving model and results...", "cyan"))
    
    # Save the trained model
    model_path = os.path.join(MODEL_DIR, "group_model.pkl")
    joblib.dump(kmeans, model_path)
    df.to_csv(GROUPED_FILE, index=False)
    
    # Save group statistics
    group_stats_file = "processed_data/group_statistics.csv"
    group_stats.to_csv(group_stats_file)
    
    # Save feature column names for later use in prediction
    feature_columns = list(df.columns.difference(['Group_ID', 'Group_Size']))
    joblib.dump(feature_columns, os.path.join(MODEL_DIR, "feature_columns.pkl"))
    
    print(colored("\n" + "=" * 60, "green"))
    print(colored("MODEL TRAINING COMPLETED!", "green", attrs=["bold"]))
    print(colored("=" * 60, "green"))
    
    print(colored("\nGroup Formation Summary:", "yellow"))
    print(f"  Total students: {len(df)}")
    print(f"  Number of groups: {df['Group_ID'].nunique()}")
    print(f"  Target group size: {STUDENTS_PER_GROUP}")
    
    # Check if all groups have 4 students
    group_sizes = df["Group_ID"].value_counts()
    balanced = all(size == STUDENTS_PER_GROUP for size in group_sizes)
    
    if balanced:
        print(colored("  ✓ All groups have exactly 4 students!", "green"))
    else:
        print(colored(f"  Groups have varying sizes: {group_sizes.unique()}", "yellow"))
    
    print(colored("\nSaved Model Files:", "yellow"))
    print(colored(f"- KMeans Model → {model_path}", "yellow"))
    print(colored(f"- Feature Scaler → {MODEL_DIR}/feature_scaler.pkl", "yellow"))
    print(colored(f"- Feature Columns → {MODEL_DIR}/feature_columns.pkl", "yellow"))
    print(colored(f"- Grouped Data → {GROUPED_FILE}", "yellow"))
    print(colored(f"- Group Statistics → {group_stats_file}", "yellow"))
    
    # Display sample groups
    print(colored("\nSample Groups (first 3 groups):", "cyan"))
    sample_groups = df[df["Group_ID"] < 3].sort_values("Group_ID")
    for group_id in sorted(sample_groups["Group_ID"].unique()):
        group_members = sample_groups[sample_groups["Group_ID"] == group_id]
        print(f"\nGroup {group_id} ({len(group_members)} members):")
        for idx, row in group_members.iterrows():
            print(f"  - {row['skills'][:30]}... | GPA: {row['gpa']} | Comm: {row['communication_score']}")
    
    return df, kmeans, scaler


if __name__ == "__main__":
    train_group_model()