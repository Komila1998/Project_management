# Adaptive Intelligence Framework for Data-Driven Optimization of Collaborative Software Project Environments

## Project Overview
Final-year undergraduate software projects often face critical challenges such as improper student group formation, poorly defined project topics, weak supervisor–student alignment, limited mentoring support, and lack of early risk detection. These issues commonly lead to unbalanced teams, project delays, repeated topic revisions, and reduced academic outcomes.
This project proposes an Adaptive Intelligence Framework as a web-based decision support system that integrates machine learning, natural language processing (NLP), and deep learning techniques to optimize collaborative software project environments. The system automates intelligent group formation, evaluates project topics using Large Language Models (LLMs), supports supervisor matching through semantic analysis, and monitors project progress using predictive analytics.
The framework aims to enhance team balance, topic feasibility, mentorship quality, and project success rates through explainable and data-driven recommendations.

## Key Objectives
Automatically form balanced student groups using machine learning based on academic performance, skills, interests, and personality traits.
Evaluate and refine student project topics using LLM-based semantic analysis.
Match students with appropriate supervisors based on topic similarity and expertise.
Provide continuous mentorship support using an AI-powered chatbot.
Detect early project risks using sentiment analysis and time-series forecasting.
Offer a centralized dashboard for academic decision support and monitoring.

## System Modules
|Module|Description|
|------|--------------|
|Intelligent Group Formation|	-Uses clustering algorithms to form balanced teams based on GPA, skills, interests, and behavioral traits.
|Topic Evaluation using LLMs|	-Analyzes project ideas for feasibility, clarity, and relevance, providing structured academic feedback.
|Supervisor–Student Matching|	-Matches project groups with supervisors using NLP-based semantic similarity analysis.
|AI-Based Mentorship Chatbot|	-Provides real-time guidance, academic support, and procedural assistance.
|Progress Evaluation & Risk Detection| -	Predicts productivity risks using sentiment analysis, anomaly detection, and LSTM models.
 
## Target Users / Beneficiaries
Undergraduate Students – Balanced teams, clear project direction, and continuous academic support.
Supervisors – Reduced workload and improved topic–student alignment.
Academic Institutions – Improved project completion rates and learning outcomes.

System Architecture
The system follows a modular web-based architecture integrating AI pipelines with a centralized database and interactive user interface.

## High-Level Workflow
Students register and create academic profiles.
The system performs intelligent group formation using clustering algorithms.
Students submit project topics for LLM-based evaluation and feedback.
NLP models match groups with suitable supervisors.
An AI chatbot provides continuous mentoring support.
Progress data is monitored to detect risks and performance issues.
Results and insights are visualized through dashboards.

## Technologies & Tools Used

##Frontend
React.js
HTML, CSS, JavaScript
Chart.js / Recharts for data visualization

## Backend
Python (Flask / FastAPI)
RESTful API architecture

## Machine Learning / AI
Python
Scikit-learn
Pandas, NumPy
K-Means Clustering (Primary model for group formation)
Gaussian Mixture Model (GMM) (Comparative model)
TF-IDF Vectorization for text feature extraction
Large Language Models (LLMs) for topic evaluation
LSTM Networks for progress and risk prediction

Model Evaluation Techniques
Silhouette Score (clustering quality)
Intra-group variance (GPA balance)
Qualitative evaluation through academic feedback

## Database
MongoDB / MySQL (based on deployment choice)
Stores student profiles, project topics, supervisor metadata, and progress logs
