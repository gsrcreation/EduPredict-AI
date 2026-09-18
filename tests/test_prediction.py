from src.data_preprocessing import validate_student_input
from src.recommendation import generate_recommendations

def test_valid_input():
    data = {"attendance":80,"study_hours":4,"assignment_score":75,
            "internal_marks":70,"previous_score":72,"backlogs":0,"participation":7}
    assert validate_student_input(data) is True

def test_invalid_attendance():
    data = {"attendance":120,"study_hours":4,"assignment_score":75,
            "internal_marks":70,"previous_score":72,"backlogs":0,"participation":7}
    try:
        validate_student_input(data)
        assert False
    except ValueError:
        assert True

def test_recommendation():
    data = {"attendance":60,"study_hours":1,"assignment_score":40,
            "internal_marks":45,"previous_score":50,"backlogs":2,"participation":3}
    recs = generate_recommendations(data, "High")
    assert len(recs) > 0