from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import (
    FacultyMember, 
    FacultyPerformance,
    CorrelationResults,
    ANOVAResult,
    CollegeEvaluationResult,
    CollegeComparisonResult,
    StatsmodelsRegressionResult,
    StatsmodelsPredictionSample,
    SklearnRegressionResult,
    SklearnPredictionSample,
)

@admin.register(FacultyMember)
class FacultyMemberAdmin(admin.ModelAdmin):
    list_display = ('name', 'position', 'sex', 'campus', 'college')
    search_fields = ('name', 'position', 'campus', 'college')
    list_filter = ('sex', 'position', 'college', 'campus')

@admin.register(FacultyPerformance)
class FacultyPerformanceAdmin(admin.ModelAdmin):
    list_display = ('faculty', 'school_year', 'semester', 'student_evaluation', 'workload_units', 'num_preparations', 'deloading')
    search_fields = ('faculty__name', 'school_year', 'semester')
    list_filter = ('semester', 'deloading')
    autocomplete_fields = ['faculty']  # optional: for easier faculty selection if many records


@admin.register(CorrelationResults)
class CorrelationResultsAdmin(admin.ModelAdmin):
    list_display = ('workload_corr', 'prep_corr', 'computed_at')


@admin.register(ANOVAResult)
class ANOVAResultAdmin(admin.ModelAdmin):
    list_display = ('factor', 'f_statistic', 'p_value', 'significant', 'created_at')


@admin.register(CollegeEvaluationResult)
class CollegeEvaluationResultAdmin(admin.ModelAdmin):
    list_display = ('college', 'average_score', 'rank')


@admin.register(CollegeComparisonResult)
class CollegeComparisonResultAdmin(admin.ModelAdmin):
    list_display = ('group1', 'group2', 'p_value', 'reject_null')


@admin.register(StatsmodelsRegressionResult)
class StatsmodelsRegressionResultAdmin(admin.ModelAdmin):
    list_display = ('r2_score', 'adjusted_r2', 'mse', 'created_at')
    readonly_fields = ('created_at',)


@admin.register(StatsmodelsPredictionSample)
class StatsmodelsPredictionSampleAdmin(admin.ModelAdmin):
    list_display = ('result', 'index', 'actual', 'predicted')


@admin.register(SklearnRegressionResult)
class SklearnRegressionResultAdmin(admin.ModelAdmin):
    list_display = ('r2_score', 'mse', 'created_at')
    readonly_fields = ('created_at',)


@admin.register(SklearnPredictionSample)
class SklearnPredictionSampleAdmin(admin.ModelAdmin):
    list_display = ('result', 'index', 'actual', 'predicted')