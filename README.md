# Adaptive Intelligence Framework for Data-Driven Optimization of Collaborative Software Project Environments

## Project Overview
Final-year undergraduate software projects often face critical challenges such as improper student group formation, poorly defined project topics, weak supervisor–student alignment, limited mentoring support, and a lack of early risk detection. These issues commonly lead to unbalanced teams, project delays, repeated topic revisions, and reduced academic outcomes.
This project proposes an Adaptive Intelligence Framework as a web-based decision support system that integrates machine learning, natural language processing (NLP), and deep learning techniques to optimize collaborative software project environments. The system automates intelligent group formation, evaluates project topics using Large Language Models (LLMs), supports supervisor matching through semantic analysis, and monitors project progress using predictive analytics.
The framework aims to enhance team balance, topic feasibility, mentorship quality, and project success rates through explainable and data-driven recommendations.

## Key Objectives
Automatically form balanced student groups using machine learning based on academic performance, skills, interests, and personality traits.
Evaluate and refine student project topics using LLM-based semantic analysis.
Match students with appropriate supervisors based on topic similarity and expertise.
Ensure fair supervisor workload distribution by considering supervisor availability and capacity during the matching process.
Provide continuous mentorship support using an AI-powered chatbot. The chatbot is implemented as a domain-specific, intent-based NLP system designed for controlled academic guidance rather than a general-purpose conversational AI.
Detect early project risks using sentiment analysis and time-series forecasting.
Offer a centralized dashboard for academic decision support and monitoring.
Ensure transparency and explainability in supervisor recommendations and academic guidance to support trust and academic decision-making.


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

## System Architecture
The system follows a modular web-based architecture integrating AI pipelines with a centralized database and interactive user interface.
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/44acf012-97bf-4085-874b-63b94cfdbbbe" />


## High-Level Workflow
1. Students register and create academic profiles.
2. The system performs intelligent group formation using clustering algorithms.
3. Students submit project topics for LLM-based evaluation and structured academic feedback.
4. NLP-based semantic analysis matches project groups with suitable supervisors based on expertise and availability.
5. A domain-specific, NLP-based chatbot provides continuous academic mentoring and procedural guidance.
6. Project progress data is monitored to detect risks and performance issues.
Results, recommendations, and alerts are visualized through interactive dashboards.

## Technologies & Tools Used

## Frontend
React.js
HTML, CSS, JavaScript
Chart.js 

## Backend
Python (Flask / FastAPI)
RESTful API architecture

## Machine Learning / AI
Python
Scikit-learn
Pandas, NumPy
K-Means Clustering (Primary model for group formation)
Gaussian Mixture Model (GMM) (Comparative model)
TF-IDF Vectorization for semantic text representation
Cosine similarity for supervisor–topic matching
Rule-based NLP techniques for domain identification
Large Language Models (LLMs) for topic evaluation
LSTM Networks for progress and risk prediction

Model Evaluation Techniques
Silhouette Score (clustering quality)
Intra-group variance (GPA balance)
Qualitative evaluation through academic feedback

## Database
MongoDB / MySQL (based on deployment choice)
Stores student profiles, project topics, supervisor metadata, and progress logs

## Admin Dashboard Features (For Your Project)
🔹 Group & Project Overview
Overview of total students, groups formed, and supervisors assigned
Group balance indicators (GPA distribution, skill diversity)
Project topic approval and evaluation status
🔹 Intelligent Group Formation Analytics
Visualization of group-wise average GPA
Skill and domain diversity per group
Communication and leadership balance analysis
🔹 Topic Evaluation & Recommendation Reports
AI-based feasibility score for each project topic
Strengths and weaknesses of proposed topics
Suggested improvements and refined topic descriptions
🔹 Supervisor Matching Dashboard
Supervisor expertise vs student topic alignment
Load balancing of supervisors
🔹 Student Progress & Risk Monitoring
Weekly progress submission tracking
Risk alerts for inactive or low-performing groups
Early warnings for project delays

## Ethical AI & Fairness (For Your Project)
🔹 Fair Group Formation
GPA stratification ensures academic balance across groups
No single group is overloaded with high or low GPA students
🔹 Bias Mitigation
Student identities anonymized during group formation
Decisions based on skills, interests, and performance metrics only
🔹 Explainable AI
Clear explanation for:
Why was a student placed in a specific group
Why a topic was accepted, rejected, or improved
Transparent rule-based and ML-based reasoning
🔹 Data Privacy & Security (Conceptual)
Secure handling of student academic records
Role-based access control (Student / Supervisor / Admin)
No exposure of sensitive data in dashboards

## Sample Outputs (For Your Project)
]🔹 Intelligent Group Formation
Group ID assignment
Group-wise average GPA
Skill diversity summary per group
🔹 Topic Evaluation Module
Topic feasibility score (0–100)
Recommendation category:
Accept
Improve
Reject
AI-generated topic refinement suggestions
🔹 Supervisor Matching
Supervisor match score
Ranked list of suitable supervisors
Final assignment recommendation
🔹 Progress & Risk Prediction
Group risk level:
Low
Medium
High
Delay probability estimate
Recommended corrective actions
