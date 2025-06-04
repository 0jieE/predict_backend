from scipy.stats import f_oneway
from performance.ml.data_loader import load_faculty_data
from performance.models import ANOVAResult


def run_anova(field_name: str):
    df = load_faculty_data()
    grouped = df.dropna(subset=['student_evaluation', field_name]).groupby(field_name)

    value_groups = []
    labels = []

    for label, group in grouped:
        evals = group['student_evaluation'].dropna()
        if len(evals) > 1:
            value_groups.append(evals)
            labels.append(label)

    if len(value_groups) < 2:
        print(f"Not enough groups in {field_name} for ANOVA.")
        return None

    f_stat, p_val = f_oneway(*value_groups)
    is_significant = p_val < 0.05

    ANOVAResult.objects.create(
        factor=field_name,
        f_statistic=f_stat,
        p_value=p_val,
        significant=is_significant
    )

    print("Saved category to database.")
    return {
        'factor': field_name,
        'f_statistic': f_stat,
        'p_value': p_val,
        'significant': is_significant,
        'labels': labels
    }

