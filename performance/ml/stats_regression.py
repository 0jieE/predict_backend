import pandas as pd
import statsmodels.api as sm

from performance.ml.data_loader import load_faculty_data
from performance.models import StatsmodelsRegressionResult, StatsmodelsPredictionSample


def run_stats_regression():
    df = load_faculty_data()

    features = ['workload_units', 'num_preparations']
    target_column = 'student_evaluation'

    # Convert to numeric
    df[features + [target_column]] = df[features + [target_column]].apply(pd.to_numeric, errors='coerce')

    # Drop rows with missing or zero values
    df = df.dropna()
    df = df[(df[features] != 0).all(axis=1)]

    if df.empty:
        raise ValueError("No data left after cleaning. Check your preprocessing logic.")

    X = df[features]
    y = df[target_column]

    # Add constant
    X_const = sm.add_constant(X, has_constant='add')

    model = sm.OLS(y, X_const).fit()
    predictions = model.predict(X_const)

    # Extract performance metrics
    r2 = model.rsquared
    adj_r2 = model.rsquared_adj
    mse = ((predictions - y) ** 2).mean()

    # Save regression result
    result_entry = StatsmodelsRegressionResult.objects.create(
        r2_score=r2,
        adjusted_r2=adj_r2,
        mse=mse,
        coefficients=model.params.to_dict(),
        p_values=model.pvalues.to_dict(),
        notes="Statsmodels linear regression run"
    )

    # Save sample predictions (first 5)
    for i in range(min(5, len(df))):
        StatsmodelsPredictionSample.objects.create(
            result=result_entry,
            index=i,
            actual=y.iloc[i],
            predicted=predictions.iloc[i]
        )

    return model, predictions, result_entry.id
