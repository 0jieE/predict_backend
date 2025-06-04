from performance.ml.data_loader import load_faculty_data
from performance.models import CorrelationResults

def correlation_analysis():
    df = load_faculty_data()
    corr = df[['student_evaluation', 'workload_units', 'num_preparations']].corr()
    target_corr = corr['student_evaluation'].drop('student_evaluation')

    workload_corr = target_corr.get('workload_units')
    prep_corr = target_corr.get('num_preparations')

    CorrelationResults.objects.create(
        workload_corr=workload_corr,
        prep_corr=prep_corr
    )

    print("Saved correlations to database.")
