def generate_recommendations(data, risk_level):
    recs = []
    if data["attendance"] < 75:
        recs.append("Improve attendance and maintain regular class participation.")
    if data["study_hours"] < 3:
        recs.append("Increase focused self-study time to at least 3 hours per day.")
    if data["assignment_score"] < 60:
        recs.append("Complete pending assignments and review missed concepts.")
    if data["internal_marks"] < 60:
        recs.append("Practice internal-exam topics and use faculty doubt sessions.")
    if data["previous_score"] < 60:
        recs.append("Revise fundamentals from previous units before advanced topics.")
    if data["backlogs"] > 0:
        recs.append("Create a backlog-clearing schedule alongside current subjects.")
    if data["participation"] < 5:
        recs.append("Participate more actively in class, quizzes, and discussions.")
    if not recs:
        recs.append("Maintain the current study routine and continue regular revision.")
    if risk_level == "High":
        recs.insert(0, "Prioritize an immediate weekly improvement plan and monitor progress.")
    elif risk_level == "Medium":
        recs.insert(0, "Set weekly targets and review performance every week.")
    return recs