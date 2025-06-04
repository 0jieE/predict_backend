from django.urls import path
from . import views
from .apiview import ManualInputView, KPIMetricsView,FacultyMemberListView

urlpatterns = [
    path('', views.landing_page, name='landing_page'),
    path('file-upload', views.upload_csv, name='upload_csv'),
    path('data-analysis/', views.data_analysis, name='start_analysis'),
    path('table/', views.performance_table, name='performance_table'),
    path("kpi-regression/", views.kpi_regression, name="kpi_regression"),
    path("analyze/<str:faculty>/", views.analyze_faculty_kpi, name="analyze_faculty_kpi"),

#api
    path("manual-input/", ManualInputView.as_view(), name="manual-input"),
    path('api/kpi-metrics/', KPIMetricsView.as_view(), name='kpi-metrics'),
    path('api/faculty/', FacultyMemberListView.as_view(), name='faculty-list'),
]
