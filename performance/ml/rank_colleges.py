import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from performance.ml.data_loader import load_faculty_data
from performance.models import CollegeEvaluationResult, CollegeComparisonResult


def rank_colleges():
    df = load_faculty_data()

    # Drop rows with missing data in required fields
    df = df.dropna(subset=['student_evaluation', 'college'])

    # Rank colleges by average student evaluation
    grouped = df.groupby('college')['student_evaluation'].mean().sort_values(ascending=False)
    ranked = grouped.reset_index().rename(columns={'student_evaluation': 'average_score'})
    ranked['rank'] = ranked['average_score'].rank(ascending=False, method='dense').astype(int)

    # Clear previous saved results
    CollegeEvaluationResult.objects.all().delete()
    CollegeComparisonResult.objects.all().delete()

    # Save rankings to DB
    for _, row in ranked.iterrows():
        CollegeEvaluationResult.objects.create(
            college=row['college'],
            average_score=row['average_score'],
            rank=row['rank']
        )

    # Perform Tukey's HSD test
    tukey = pairwise_tukeyhsd(endog=df['student_evaluation'], groups=df['college'], alpha=0.05)
    summary_data = tukey.summary().data[1:]  # skip header row

    for row in summary_data:
        CollegeComparisonResult.objects.create(
            group1=row[0],
            group2=row[1],
            p_value=row[4],
            reject_null=str(row[5]).strip().lower() == 'true'
        )
    print("Saved rankig to database.")
    return {
        'ranking': ranked.to_dict(orient='records'),
        'tukey_summary': summary_data
    }
