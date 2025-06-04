import pandas as pd
import numpy as np
import statsmodels.api as sm
import io
import chardet
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from django.shortcuts import render
import tensorflow as tf
import numpy as np


def landing_page(request):
    filename = request.session.get("csv_filename")
    return render(request, 'performance/landing.html', {"csv_filename": filename})

@csrf_exempt
def upload_csv(request):
    if request.method == "POST" and request.FILES.get("csv"):
        csv_file = request.FILES["csv"]
        file_bytes = csv_file.read()
        encoding = chardet.detect(file_bytes)["encoding"] or "utf-8"
        try:
            decoded_file = file_bytes.decode(encoding)
        except Exception as e:
            return render(request, "error.html", {"message": f"Error decoding file: {e}"})

        # Read CSV skipping the first row (header grouping)
        df = pd.read_csv(io.StringIO(decoded_file), skiprows=1)

        # Clean column names
        df.columns = [col.strip() for col in df.columns]
        print(df.columns.tolist())
        # Save full dataframe JSON in session (includes column names and data)
        request.session["csv_data_json"] = df.to_json(orient="records")  # list of dicts as JSON string

        # Also save headers and rows separately (optional)
        request.session["csv_headers"] = df.columns.tolist()
        request.session["csv_rows"] = df.values.tolist()
        request.session["csv_filename"] = csv_file.name

        print(df.columns.tolist())

        return redirect("landing_page")

    return render(request, "error.html", {"message": "No file uploaded."})


def data_analysis(request):
    return render(request, 'performance/data_analysis.html')

def performance_table(request):
    headers = request.session.get("csv_headers", [])
    rows = request.session.get("csv_rows", [])

    if not headers or not rows:
        # If no data in session, return an empty table or error message
        return render(request, "performance/_table.html", {
            "headers": [],
            "rows": [],
        })

    # Try to find index of "FACULTY NAME"
    try:
        faculty_index = headers.index("NAME")  # Correct label here
    except ValueError:
        faculty_index = None


    # Annotate each row with whether it should show the Analyze button
    displayed_faculty = set()
    annotated_rows = []
    for row in rows:
        faculty = row[faculty_index] if len(row) > faculty_index else None
        show_button = False
        if faculty and faculty not in displayed_faculty:
            show_button = True
            displayed_faculty.add(faculty)
        annotated_rows.append((row, show_button, faculty))
    return render(request, "performance/_table.html", {
        "headers": headers,
        "rows": annotated_rows,
    })


def kpi_regression(request):
    json_data = request.session.get("csv_data_json")
    if not json_data:
        return render(request, "error.html", {"message": "No uploaded data found."})

    try:
        # Wrap json_data string in StringIO to avoid FutureWarning
        data_list = pd.read_json(io.StringIO(json_data), orient="records")
    except Exception as e:
        return render(request, "error.html", {"message": f"Error parsing stored data: {e}"})

    df = data_list

    try:
        # Convert columns to numeric and handle non-numeric data (e.g., study leave)
        df["NO. OF WORKLOAD (UNIT).1"] = pd.to_numeric(df["NO. OF WORKLOAD (UNIT).1"], errors="coerce")
        df["NO. OF PREPARATIONS.1"] = pd.to_numeric(df["NO. OF PREPARATIONS.1"], errors="coerce")
        df["STUDENT EVALUATIONS.1"] = pd.to_numeric(df["STUDENT EVALUATIONS.1"], errors="coerce")
        df["DELOADING_BINARY"] = df["DELOADING.1"].map({"Yes": 1, "No": 0})
    except KeyError as e:
        return render(request, "error.html", {"message": f"Missing column in data: {e}"})
    except Exception as e:
        return render(request, "error.html", {"message": f"Data formatting error: {e}"})

    # Prepare dependent and independent variables
    y = df["STUDENT EVALUATIONS.1"]
    X = df[["NO. OF WORKLOAD (UNIT).1", "NO. OF PREPARATIONS.1"]]

    # Drop rows with NaN or infinite values in X or y
    valid_mask = (~X.isnull().any(axis=1)) & (~y.isnull())
    X_clean = X[valid_mask].copy()
    y_clean = y[valid_mask].copy()

    finite_mask = np.isfinite(X_clean).all(axis=1) & np.isfinite(y_clean)
    X_clean = X_clean[finite_mask]
    y_clean = y_clean[finite_mask]

    # Add constant (intercept)
    X_clean = sm.add_constant(X_clean)

    # Fit regression model
    model = sm.OLS(y_clean, X_clean).fit()
    summary = model.summary().as_text()

    pvalues = model.pvalues.to_dict()
    conf_int = model.conf_int()
    coef_interpretation = {}

    for feature, coef in model.params.items():
        if feature == 'const':
            continue
        pval = pvalues.get(feature, None)
        ci_low, ci_high = conf_int.loc[feature] if feature in conf_int.index else (None, None)
        signif = "significant" if pval is not None and pval < 0.05 else "not significant"
        if coef > 0:
            interp = f"A unit increase in {feature} is associated with an increase of {coef:.3f} in Student Evaluations."
        elif coef < 0:
            interp = f"A unit increase in {feature} is associated with a decrease of {abs(coef):.3f} in Student Evaluations."
        else:
            interp = f"{feature} appears to have no effect on Student Evaluations."

        coef_interpretation[feature] = {
            "coef": coef,
            "pval": pval,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "significance": signif,
            "interpretation": interp,
        }

    return render(request, "performance/kpi_regression.html", {
        "summary": summary,
        "r_squared": round(model.rsquared, 3),
        "coefficients": model.params.to_dict(),
        "coef_details": coef_interpretation,
    })


def prepare_data(json_data):
    df = pd.read_json(io.StringIO(json_data), orient="records")
    df["NAME"] = df["NAME"].str.strip().str.upper()
    
    records = []
    for _, row in df.iterrows():
        base_data = {
            "NAME": row["NAME"],
            "SEX": row["SEX"],
            "RANK": row.get("POSITION", None),
        }
        # Semester 1
        records.append({
            **base_data,
            "STUDENT_EVAL": row["STUDENT EVALUATIONS"],
            "WORKLOAD": row["NO. OF WORKLOAD (UNIT)"],
            "PREP": row["NO. OF PREPARATIONS"],
            "DELOADING": row["DELOADING"]
        })
        # Semester 2
        records.append({
            **base_data,
            "STUDENT_EVAL": row["STUDENT EVALUATIONS.1"],
            "WORKLOAD": row["NO. OF WORKLOAD (UNIT).1"],
            "PREP": row["NO. OF PREPARATIONS.1"],
            "DELOADING": row["DELOADING.1"]
        })

    long_df = pd.DataFrame(records)

    # Cleaning helper
    def clean_float(val):
        try:
            return float(str(val).replace("..", "."))
        except ValueError:
            return None

    for col in ["STUDENT_EVAL", "WORKLOAD", "PREP"]:
        long_df[col] = long_df[col].apply(clean_float)
    long_df = long_df.dropna(subset=["STUDENT_EVAL", "WORKLOAD", "PREP", "DELOADING", "SEX"])

    # Binary encoding deloading
    long_df["DELOADING_BINARY"] = long_df["DELOADING"].map({"Yes": 1, "No": 0})

    # Label encode categorical features
    le_sex = LabelEncoder()
    le_rank = LabelEncoder()

    long_df["SEX_ENC"] = le_sex.fit_transform(long_df["SEX"])
    long_df["RANK_ENC"] = le_rank.fit_transform(long_df["RANK"].fillna("Unknown"))

    return long_df, le_sex, le_rank

def build_and_train_model(X_train, y_train, X_val, y_val, epochs=50):
    input_dim = X_train.shape[1]
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Regression output
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), verbose=0)
    return model, history

def predict_faculty_performance(model, X):
    preds = model.predict(X).flatten()
    return preds.tolist()

def analyze_faculty_kpi(request, faculty):
    json_data = request.session.get("csv_data_json")
    if not json_data:
        return HttpResponse("No data uploaded", status=400)

    faculty = faculty.strip().upper()
    long_df, le_sex, le_rank = prepare_data(json_data)

    faculty_df = long_df[long_df["NAME"] == faculty]
    if faculty_df.empty:
        return HttpResponse("No records for this faculty", status=404)

    X = long_df[["WORKLOAD", "PREP", "DELOADING_BINARY", "SEX_ENC", "RANK_ENC"]]
    y = long_df["STUDENT_EVAL"].astype(float)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model, history = build_and_train_model(X_train, y_train, X_val, y_val)

    # Prepare faculty data for prediction
    faculty_X = faculty_df[["WORKLOAD", "PREP", "DELOADING_BINARY", "SEX_ENC", "RANK_ENC"]]
    faculty_y_true = faculty_df["STUDENT_EVAL"].astype(float).tolist()
    faculty_y_pred = predict_faculty_performance(model, faculty_X)

    if len(faculty_y_true) > 1:
        # Simple linear trend (slope) calculation
        x = np.arange(len(faculty_y_true))
        y = np.array(faculty_y_true)
        slope = np.polyfit(x, y, 1)[0]  # slope of linear fit

        if slope > 0.05:
            trend = "Improving"
        elif slope < -0.05:
            trend = "Declining"
        else:
            trend = "Stable"
    else:
        trend = "Not enough data to determine trend"

    per_semester = [{
        "semester": i+1,
        "actual": round(faculty_y_true[i], 2),
        "predicted": round(faculty_y_pred[i], 2),
        "difference": round(faculty_y_true[i] - faculty_y_pred[i], 2),
    } for i in range(len(faculty_y_true))]

    summary = {
        "Faculty": faculty,
        "Semesters Counted": len(faculty_df),
        "Avg Workload": round(faculty_df["WORKLOAD"].mean(), 2),
        "Avg Preparation": round(faculty_df["PREP"].mean(), 2),
        "Avg Student Evaluation": round(faculty_df["STUDENT_EVAL"].mean(), 2),
        "Deloading Occurrences": int(faculty_df["DELOADING_BINARY"].sum()),
        "Trend": trend,
    }

    context = {
        "summary": summary,
        "predictions": per_semester,
        "r_squared": round(model.evaluate(X_val, y_val, verbose=0)[1], 3),  # MAE as a metric, adjust as needed
    }
    return render(request, "performance/_faculty_summary.html", context)

    