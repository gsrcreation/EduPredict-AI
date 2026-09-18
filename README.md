# EduPredict AI
## Student Performance Prediction & Early Intervention System

### Overview
EduPredict AI is an academic AI/ML project that demonstrates an end-to-end pipeline for student performance prediction, risk classification, analytics, and personalized intervention recommendations.

### Features
- Student data validation and management
- Data preprocessing and EDA
- Regression-based performance prediction
- Classification-based risk prediction
- Model comparison and evaluation
- Rule-based personalized recommendations
- Streamlit analytics dashboard
- SQLite storage for saved predictions
- Automated tests

### Technologies
Python, Pandas, NumPy, Scikit-learn, Streamlit, Matplotlib, SQLite, Joblib, Pytest.

### Dataset
`data/student_data.csv` is a synthetic dataset generated for academic demonstration. It contains 500 records. If the instructor requires an externally sourced dataset, replace it with an approved dataset and update the dataset description/results.

### Installation
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python -m src.train_model
streamlit run app.py
```

### Testing
```bash
pytest -q
```

### Project Structure
See the repository folders for modular preprocessing, training, prediction, recommendation, database, EDA, tests, and UI code.

### ML Models
Regression: Linear Regression and Random Forest Regressor.
Classification: Logistic Regression and Random Forest Classifier.
The training script selects the regression model with the lowest RMSE and the classifier with the highest weighted F1 score.

### Academic Note
This project is intended for educational evaluation. Predictions should not be treated as definitive judgments about real students. The synthetic dataset is used to make the repository self-contained.
