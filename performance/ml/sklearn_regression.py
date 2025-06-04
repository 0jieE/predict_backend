import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from performance.ml.data_loader import load_faculty_data
from performance.models import SklearnRegressionResult, SklearnPredictionSample


def run_sklearn_regression():
    df = load_faculty_data()

    # Drop rows with missing or zero workload/num_preparations
    df = df.dropna(subset=['student_evaluation', 'workload_units', 'num_preparations', 'sex', 'deloading'])
    df = df[(df['workload_units'] > 0) & (df['num_preparations'] > 0)]

    df['deloading'] = df['deloading'].astype(int)

    X = df[['workload_units', 'num_preparations', 'deloading', 'sex']]
    y = df['student_evaluation']

    # One-hot encode 'sex'
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(drop='first'), ['sex'])],
        remainder='passthrough'
    )

    model = Pipeline([
        ('preprocess', preprocessor),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Extract feature names after transformation
    transformed_feature_names = (
        model.named_steps['preprocess']
        .transformers_[0][1]  # OneHotEncoder
        .get_feature_names_out(['sex']).tolist() +
        ['workload_units', 'num_preparations', 'deloading']
    )

    coefficients = model.named_steps['regressor'].coef_.round(4)

    # Save regression result to DB
    result_entry = SklearnRegressionResult.objects.create(
        r2_score=r2_score(y_test, predictions),
        mse=mean_squared_error(y_test, predictions),
        coefficients=dict(zip(transformed_feature_names, coefficients)),
        notes="Sklearn linear regression run"
    )

    # Save sample predictions (first 5)
    for i in range(min(5, len(y_test))):
        SklearnPredictionSample.objects.create(
            result=result_entry,
            index=i,
            actual=y_test.iloc[i],
            predicted=predictions[i]
        )
    print("Saved successfully to database.")
    return {
        'model': model,
        'r2_score': result_entry.r2_score,
        'mse': result_entry.mse,
        'predictions': predictions,
        'true': y_test,
        'feature_names': transformed_feature_names,
        'db_entry_id': result_entry.id
    }
