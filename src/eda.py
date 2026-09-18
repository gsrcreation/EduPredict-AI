import pandas as pd
import matplotlib.pyplot as plt

def summary(df):
    return df.describe(include="all")

def save_plots(df, out_dir="screenshots"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(7,4))
    df["final_score"].hist()
    plt.title("Final Score Distribution")
    plt.xlabel("Final Score")
    plt.ylabel("Students")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/score_distribution.png")
    plt.close()

    plt.figure(figsize=(7,4))
    plt.scatter(df["attendance"], df["final_score"])
    plt.title("Attendance vs Final Score")
    plt.xlabel("Attendance (%)")
    plt.ylabel("Final Score")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/attendance_vs_score.png")
    plt.close()