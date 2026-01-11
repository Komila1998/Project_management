## Intelligent Group Formation & Project Topic Evaluation System

Final Year Research Project – PP1 (Checklist 1)

## Project Overview

This component focuses on developing an Intelligent Group Formation and Early Project Topic Evaluation System for undergraduate software engineering projects.
Unlike traditional manual project allocation methods, this system uses machine learning and large language models (LLMs) to:
Automatically form balanced student project groups
Analyze student skills, interests, and academic attributes
Ensure equal group size with fairness constraints
Evaluate project topics early with structured academic feedback
Reduce free-rider issues and supervisor workload
Due to the lack of structured datasets for student grouping and topic feasibility assessment, this component adopts a hybrid classical ML + rule-based + LLM approach that is transparent, explainable, and suitable for real academic environments.

## Main Objectives
Automatically form balanced student groups using ML
Extract and combine textual and numerical student features
Ensure fixed group size (4 students) with custom balancing logic
Reduce free-rider and imbalance issues in student teams
Evaluate project topics at an early stage using LLMs
Provide structured, explainable, and actionable academic feedback
Support scalable and reusable group formation for large cohorts
Maintain clean, modular, and version-controlled development

## System Architecture
🔹 High-Level Architecture (Conceptual)
┌────────────────────────────┐
│        Student Input       │
│ (Profile + Skills + Topic) │
└─────────────┬──────────────┘
              │
┌─────────────▼──────────────┐
│ Data Preprocessing Module  │
│ (Cleaning & Text Merging)  │
└─────────────┬──────────────┘
              │
┌─────────────▼──────────────┐
│ Feature Engineering        │
│ TF-IDF + Numeric Scaling   │
└─────────────┬──────────────┘
              │
┌─────────────▼──────────────┐
│ ML Group Formation Model   │
│ (K-Means + Balancing Logic)│
└─────────────┬──────────────┘
              │
┌─────────────▼──────────────┐
│ LLM Topic Evaluation       │
│ (Score + Feedback)         │
└─────────────┬──────────────┘
              │
┌─────────────▼──────────────┐
│ Final Outputs              │
│ Groups + Scores + Feedback │
└────────────────────────────┘

## Models & Logic Used
## Model 1: Student Feature Engineering Module

Type: Classical feature extraction (Non-ML)
Purpose: Convert student profiles into numerical representations
Features extracted: GPA, Availability hours, Communication score, Leadership score, TF-IDF vectors of skills, interests, experience, and domain

## Why used:
Low data dependency
High interpretability
Suitable for unsupervised grouping

## Model: K-Means Group Formation Model (Predefined ML)
Algorithm: K-Means Clustering (scikit-learn)
Input: Combined numerical + TF-IDF features
Output: Similarity-based student clusters

## Role:
Identify students with similar interests and skills
Guide intelligent group formation
Note:
K-Means is used only for similarity detection, not final group assignment.

## Model 3: Custom Group Balancing Logic (Core Contribution)
Type: Rule-based optimization logic
Purpose: Ensure exactly 4 students per group and balanced composition
Logic applied: Calculate the required number of groups, Sort students by similarity, Assign students sequentially into groups of 4, Redistribute leftover students to the smallest groups, Validate group size and balance metrics

## Why needed:
Standard clustering algorithms do not guarantee equal group sizes.
## Model 4: LLM-Based Project Topic Evaluation Module
Type: Large Language Model (LLM)
Purpose: Evaluate project topics at an early stage
Outputs generated: Clarity, Feasibility, Novelty, Risks, Improvement suggestions, Overall topic score (0–100), Decision (Accept / Revise / Reject)

## Why used:
No automated academic topic evaluation datasets exist
Provides early guidance and reduces supervisor intervention

## Project Folder Structure
Intelligent_Group_Formation_System/
│
├── Dataset/
│   └── Dataset.csv              # Raw student data
│
├── processed_data/
│   ├── processed_students.csv   # Cleaned student data
│   ├── grouped_students.csv     # Final group assignments
│   └── group_statistics.csv     # Group-level metrics
│
├── models/
│   ├── group_model.pkl          # Trained K-Means model
│   ├── tfidf_vectorizer.pkl     # TF-IDF model
│   ├── scaler.pkl               # Feature scaler
│
├── Result/
│   └── predicted_groups.csv     # Groups + topic feedback
│
├── data_preparation.py
├── feature_engineering.py
├── model_training.py
├── model_predict.py
├── Final_predict_evaluation_run.py
│
├── README.md
└── requirements.txt

## Script Responsibilities (My Component)
File	Purpose
data_preparation.py	Data cleaning & text feature creation
feature_engineering.py	TF-IDF + numeric feature generation
model_training.py	K-Means training + group balancing
model_predict.py	Group prediction for new students
Final_predict_evaluation_run.py	End-to-end grouping + topic evaluation

## Dependencies
Core Libraries
Python 3.9+
NumPy
Pandas
Scikit-learn
Joblib
OpenAI API (LLM)
