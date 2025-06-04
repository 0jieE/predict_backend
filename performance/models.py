from django.db import models

# models.py

class FacultyMember(models.Model):
    name = models.CharField(max_length=100)
    position = models.CharField(max_length=100)
    sex = models.CharField(max_length=10)
    campus = models.CharField(max_length=50)
    college = models.CharField(max_length=50)

    def __str__(self):
        return self.name


class FacultyPerformance(models.Model):
    SEMESTER_CHOICES = [
        ('1st Sem', '1st Sem'),
        ('2nd Sem', '2nd Sem'),
    ]
    faculty = models.ForeignKey(FacultyMember, on_delete=models.CASCADE)
    school_year = models.CharField(max_length=25, null=True, blank=True)
    semester = models.CharField(max_length=25, choices=SEMESTER_CHOICES)

    student_evaluation = models.FloatField(null=True, blank=True)
    workload_units = models.FloatField(null=True, blank=True)
    num_preparations = models.IntegerField(null=True, blank=True)
    deloading = models.BooleanField()

    def __str__(self):
        return f"{self.faculty.name} - {self.semester}"

class CorrelationResults(models.Model):
    workload_corr = models.FloatField(null=True, blank=True)
    prep_corr = models.FloatField(null=True, blank=True)
    computed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Correlation @ {self.computed_at.strftime('%Y-%m-%d %H:%M')}"

class ANOVAResult(models.Model):
    factor = models.CharField(max_length=100)  # e.g., "college", "sex", etc.
    f_statistic = models.FloatField()
    p_value = models.FloatField()
    significant = models.BooleanField()
    created_at = models.DateTimeField(auto_now_add=True)

class CollegeEvaluationResult(models.Model):
    college = models.CharField(max_length=100)
    average_score = models.FloatField()
    rank = models.PositiveIntegerField()

    class Meta:
        ordering = ['rank']

class CollegeComparisonResult(models.Model):
    group1 = models.CharField(max_length=100)
    group2 = models.CharField(max_length=100)
    p_value = models.FloatField()
    reject_null = models.BooleanField()  # True if significant difference (p < 0.05)


class StatsmodelsRegressionResult(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    r2_score = models.FloatField(null=True, blank=True)
    adjusted_r2 = models.FloatField(null=True, blank=True)
    mse = models.FloatField(null=True, blank=True)
    coefficients = models.JSONField(null=True, blank=True)
    p_values = models.JSONField(null=True, blank=True)
    notes = models.TextField(blank=True)


class StatsmodelsPredictionSample(models.Model):
    result = models.ForeignKey(StatsmodelsRegressionResult, on_delete=models.CASCADE, related_name='samples')
    index = models.IntegerField(null=True, blank=True)
    actual = models.FloatField(null=True, blank=True)
    predicted = models.FloatField(null=True, blank=True)


class SklearnRegressionResult(models.Model):
    r2_score = models.FloatField(null=True, blank=True)
    mse = models.FloatField(null=True, blank=True)
    coefficients = models.JSONField(null=True, blank=True)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class SklearnPredictionSample(models.Model):
    result = models.ForeignKey(SklearnRegressionResult, on_delete=models.CASCADE, related_name='samples')
    index = models.IntegerField(null=True, blank=True)
    actual = models.FloatField(null=True, blank=True)
    predicted = models.FloatField(null=True, blank=True)
