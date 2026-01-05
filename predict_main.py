import pandas as pd
from predict import StudentSupervisorMatcher
from config import STUDENT_FILE, SUPERVISOR_FILE

def main():
    students_df = pd.read_csv(STUDENT_FILE)
    supervisors_df = pd.read_csv(SUPERVISOR_FILE)
    
    matcher = StudentSupervisorMatcher()
    results = matcher.process_students(students_df, supervisors_df)
    matcher.save_results(results)

if __name__ == "__main__":
    main()
