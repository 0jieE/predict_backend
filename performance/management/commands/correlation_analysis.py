from django.core.management.base import BaseCommand
from performance.ml.data_loader import load_faculty_data

class Command(BaseCommand):
    help = "Run correlation analysis on faculty performance data"

    def handle(self, *args, **kwargs):
        import seaborn as sns
        import matplotlib.pyplot as plt

        df = load_faculty_data()
        numeric_df = df[["student_evaluation", "workload_units", "num_preparations", "deloading"]].dropna()

        print("Correlation matrix:")
        print(numeric_df.corr())

        # Optional: plot heatmap
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
        plt.title("Faculty KPI Correlation Heatmap")
        plt.show()
