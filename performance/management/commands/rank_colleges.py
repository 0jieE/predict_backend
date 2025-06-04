from django.core.management.base import BaseCommand
from performance.ml.data_loader import load_faculty_data
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

class Command(BaseCommand):
    help = "Ranks colleges by average student evaluation and runs Tukey's HSD test."

    def handle(self, *args, **kwargs):
        df = load_faculty_data()

        # Drop missing evaluations
        df = df.dropna(subset=['student_evaluation', 'college'])

        # Step 1: Rank colleges by mean evaluation score
        college_means = df.groupby('college')['student_evaluation'].mean().sort_values(ascending=False)
        print("\nMean Evaluation Scores by College (Highest to Lowest):")
        print(college_means)

        # Step 2: Tukey's HSD Test
        print("\nTukey's HSD Test Results:")
        tukey = pairwise_tukeyhsd(
            endog=df['student_evaluation'],
            groups=df['college'],
            alpha=0.05
        )
        print(tukey.summary())
