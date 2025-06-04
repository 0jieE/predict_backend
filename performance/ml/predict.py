import numpy as np
from joblib import load
from tensorflow.keras.models import load_model # type: ignore
import os

# Load paths
current_dir = os.path.dirname(__file__)
preprocessor = load(os.path.join(current_dir, "preprocessor.joblib"))
model = load_model(os.path.join(current_dir, "faculty_model.keras"))

def predict_performance(input_data_dict):
    """
    Expects a dict with same keys as your training features
    Returns predicted student evaluation score
    """
    # Convert dict to DataFrame
    import pandas as pd
    df = pd.DataFrame([input_data_dict])

    # Transform input using the same pipeline
    X = preprocessor.transform(df)

    # Predict
    prediction = model.predict(X)
    return prediction[0][0]  # Return scalar

sample_input = {
     "workload_units": 18,
     "num_preparations": 3,
     "deloading": 0,
     "sex": "M",
     "position": "Associate Professor",
     "campus": "main",
     "college": "CASS",
     "school_year": "2023-2024",
     "semester": "1st Sem"
}