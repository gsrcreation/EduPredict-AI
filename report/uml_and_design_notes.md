# UML and Design Notes

## Use Case Diagram
Actors: Student, Faculty/Academic Mentor, Administrator.

Student: Enter data, View prediction, View recommendations, View analytics.
Faculty/Mentor: Review student records, View risk distribution, Review predictions.
Administrator: Manage dataset, train/evaluate models, manage records.

## Component/Class Diagram
StudentData -> DataPreprocessor -> PredictionEngine -> MLModel
PredictionEngine -> RecommendationEngine
StudentData -> DatabaseManager
Analytics -> Dashboard

## Sequence
User -> Streamlit UI -> Validator -> Preprocessor -> Prediction Engine -> ML Model -> Recommendation Engine -> UI -> Database.

## ER Design
STUDENT(student_id PK, name, program, semester, attendance, study_hours, assignment_score, internal_marks, previous_score, backlogs, participation, predicted_score, risk_level)
