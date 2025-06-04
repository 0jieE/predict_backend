import pandas as pd
from performance.models import FacultyPerformance


def load_faculty_data():
    # Load all FacultyPerformance records, including related FacultyMember data
    queryset = FacultyPerformance.objects.select_related("faculty").all()

    data = []
    for record in queryset:
        # Skip records with missing target or invalid predictors
        if (
            record.student_evaluation is None or
            record.workload_units == 0 or
            record.num_preparations == 0
        ):
            continue

        data.append({
            # Target variable
            "student_evaluation": record.student_evaluation,

            # Numeric predictors
            "workload_units": record.workload_units,
            "num_preparations": record.num_preparations,
            "deloading": int(record.deloading) if record.deloading is not None else None,

            # Categorical predictors (for future encoding)
            "sex": record.faculty.sex,
            "position": record.faculty.position,
            "campus": record.faculty.campus,
            "college": record.faculty.college,
            "school_year": record.school_year,
            "semester": record.semester,
        })

    df = pd.DataFrame(data)
    return df
