import streamlit as st
import pandas as pd
from src.data_preprocessing import load_data
from src.prediction import predict_student
from src.recommendation import generate_recommendations
from src.database import init_db, add_student, get_students

st.set_page_config(page_title="EduPredict AI", page_icon="🎓", layout="wide")
init_db()

st.title("🎓 EduPredict AI")
st.caption("Student Performance Prediction & Early Intervention System")

df = load_data()

tabs = st.tabs(["🏠 Dashboard", "🔮 Prediction", "📊 Analytics", "👥 Student Records", "ℹ️ About"])

with tabs[0]:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Students", len(df))
    c2.metric("Average Score", f"{df.final_score.mean():.1f}")
    c3.metric("Avg Attendance", f"{df.attendance.mean():.1f}%")
    c4.metric("High Risk", int((df.risk_level=="High").sum()))
    st.subheader("Risk Distribution")
    st.bar_chart(df["risk_level"].value_counts())
    st.info("Use the Prediction tab to generate a prediction and intervention recommendations for a student.")

with tabs[1]:
    st.subheader("Student Prediction")
    with st.form("prediction_form"):
        name = st.text_input("Student Name", "Demo Student")
        program = st.selectbox("Program", ["AI","AIML","CSE","ECE"])
        semester = st.selectbox("Semester", [1,2,3,4])
        a,b,c = st.columns(3)
        attendance = a.number_input("Attendance (%)", 0.0, 100.0, 80.0)
        study_hours = b.number_input("Study Hours/Day", 0.0, 24.0, 3.5)
        assignment = c.number_input("Assignment Score", 0.0, 100.0, 75.0)
        d,e,f = st.columns(3)
        internal = d.number_input("Internal Marks", 0.0, 100.0, 70.0)
        previous = e.number_input("Previous Score", 0.0, 100.0, 72.0)
        backlogs = f.number_input("Backlogs", 0, 20, 0)
        participation = st.number_input("Participation (1-10)", 0.0, 10.0, 7.0)
        submitted = st.form_submit_button("Predict Performance")
    if submitted:
        data = {"attendance":attendance,"study_hours":study_hours,"assignment_score":assignment,
                "internal_marks":internal,"previous_score":previous,"backlogs":backlogs,
                "participation":participation}
        try:
            result = predict_student(data)
            x,y,z = st.columns(3)
            x.metric("Predicted Score", f"{result['predicted_score']:.1f}/100")
            y.metric("Risk Level", result["risk_level"])
            z.metric("Confidence", "N/A" if result["confidence"] is None else f"{result['confidence']*100:.1f}%")
            st.subheader("Personalized Recommendations")
            for r in generate_recommendations(data, result["risk_level"]):
                st.write("•", r)
            add_student([name,program,semester,attendance,study_hours,assignment,internal,previous,
                         backlogs,participation,result["predicted_score"],result["risk_level"]])
            st.success("Prediction saved to student records.")
        except Exception as ex:
            st.error(str(ex))

with tabs[2]:
    st.subheader("Dataset Analytics")
    st.dataframe(df.head(20), use_container_width=True)
    st.subheader("Average Score by Program")
    st.bar_chart(df.groupby("program")["final_score"].mean())
    st.subheader("Attendance vs Final Score")
    st.scatter_chart(df[["attendance","final_score"]].rename(columns={"attendance":"Attendance","final_score":"Score"}))
    st.subheader("Model Results")
    import os
    if os.path.exists("models/regression_results.csv"):
        st.write("Regression")
        st.dataframe(pd.read_csv("models/regression_results.csv"), use_container_width=True)
    if os.path.exists("models/classification_results.csv"):
        st.write("Classification")
        st.dataframe(pd.read_csv("models/classification_results.csv"), use_container_width=True)

with tabs[3]:
    st.subheader("Saved Prediction Records")
    records = get_students()
    cols = ["ID","Name","Program","Semester","Attendance","Study Hours","Assignment",
            "Internal","Previous","Backlogs","Participation","Predicted Score","Risk"]
    st.dataframe(pd.DataFrame(records, columns=cols), use_container_width=True)

with tabs[4]:
    st.subheader("About EduPredict AI")
    st.write("""
    EduPredict AI demonstrates an end-to-end machine learning workflow:
    data preparation, exploratory analysis, regression, classification,
    model evaluation, prediction, recommendations, and record storage.
    """)
    st.write("**Academic note:** The included dataset is synthetic and is intended for demonstration/testing. Replace it with an approved real dataset if your instructor requires one.")