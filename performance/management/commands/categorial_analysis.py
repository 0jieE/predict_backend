from django.core.management.base import BaseCommand
from performance.ml.data_loader import load_faculty_data
from scipy.stats import f_oneway

class Command(BaseCommand):
    help = 'Performs one-way ANOVA on student evaluation scores across colleges.'

    def handle(self, *args, **kwargs):
        df = load_faculty_data()
        grouped = df.dropna(subset=['student_evaluation', 'college']).groupby('college')

        college_groups = []
        labels = []

        for college, group in grouped:
            evals = group['student_evaluation'].dropna()
            if len(evals) > 1:  # require at least 2 values for ANOVA
                college_groups.append(evals)
                labels.append(college)

        if len(college_groups) < 2:
            self.stdout.write(self.style.WARNING("Not enough college groups with data for ANOVA."))
            return

        f_stat, p_val = f_oneway(*college_groups)

        self.stdout.write(f"\nANOVA Results Across Colleges:")
        self.stdout.write(f"  F-statistic: {f_stat:.4f}")
        self.stdout.write(f"  P-value: {p_val:.4f}")

        if p_val < 0.05:
            self.stdout.write(self.style.NOTICE("  Significant difference in student evaluations between colleges (p < 0.05)"))
        else:
            self.stdout.write("  No significant difference in student evaluations between colleges (p ≥ 0.05)")
