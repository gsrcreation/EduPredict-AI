import sqlite3
from pathlib import Path

DB_PATH = Path("data/edupredict.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS students(
        student_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        program TEXT,
        semester INTEGER,
        attendance REAL,
        study_hours REAL,
        assignment_score REAL,
        internal_marks REAL,
        previous_score REAL,
        backlogs INTEGER,
        participation REAL,
        predicted_score REAL,
        risk_level TEXT
    )""")
    conn.commit()
    conn.close()

def add_student(record):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO students
    (name,program,semester,attendance,study_hours,assignment_score,internal_marks,
     previous_score,backlogs,participation,predicted_score,risk_level)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", tuple(record))
    conn.commit()
    conn.close()

def get_students():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT * FROM students ORDER BY student_id DESC").fetchall()
    conn.close()
    return rows