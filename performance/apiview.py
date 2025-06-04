from rest_framework.views import APIView
from rest_framework.generics import ListAPIView
from rest_framework.response import Response
from rest_framework import status
from .models import (
    CorrelationResults, ANOVAResult, CollegeEvaluationResult,FacultyMember, FacultyPerformance,
    CollegeComparisonResult, StatsmodelsRegressionResult, SklearnRegressionResult
)
from .serializers import FacultyPerformanceSerializer, FacultyMemberSerializer
from rest_framework.decorators import api_view
from django.http import JsonResponse


class FacultyMemberListView(ListAPIView):
    queryset = FacultyMember.objects.all()
    serializer_class = FacultyMemberSerializer

class PerformanceAPIView(ListAPIView):
    queryset = FacultyPerformance.objects.select_related("faculty").all()
    serializer_class = FacultyPerformanceSerializer



class ManualInputView(APIView):
    def post(self, request):
        faculty_data = request.data.get('faculty')
        performance_data = request.data.get('performance')

        if not faculty_data or not performance_data:
            return Response({'error': 'Both faculty and performance data are required.'}, status=status.HTTP_400_BAD_REQUEST)

        # Try to get faculty by id or name (case-insensitive)
        faculty_instance = None
        faculty_id = faculty_data.get('id')
        faculty_name = faculty_data.get('name', '').strip()

        if faculty_id:
            faculty_instance = FacultyMember.objects.filter(id=faculty_id).first()
        elif faculty_name:
            faculty_instance = FacultyMember.objects.filter(name__iexact=faculty_name).first()

        if not faculty_instance:
            # Create new FacultyMember
            faculty_serializer = FacultyMemberSerializer(data=faculty_data)
            if faculty_serializer.is_valid():
                faculty_instance = faculty_serializer.save()
            else:
                return Response(faculty_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        # Add faculty id to performance data for FK relation
        performance_data['faculty'] = faculty_instance.id

        performance_serializer = FacultyPerformanceSerializer(data=performance_data)
        if performance_serializer.is_valid():
            performance_serializer.save()
            return Response({'status': 'Entry created'}, status=status.HTTP_201_CREATED)
        else:
            return Response(performance_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class KPIMetricsView(APIView):
    def get(self, request):
        response_data = {}

        # Correlation
        correlation = CorrelationResults.objects.last()
        if correlation:
            response_data['correlation'] = {
                'workload_corr': correlation.workload_corr,
                'prep_corr': correlation.prep_corr,
                'computed_at': correlation.computed_at,
            }

        # ANOVA (college & sex latest)
        response_data['anova'] = {}
        for factor in ['college', 'sex']:
            result = ANOVAResult.objects.filter(factor=factor).last()
            if result:
                response_data['anova'][factor] = {
                    'f_statistic': result.f_statistic,
                    'p_value': result.p_value,
                    'significant': result.significant,
                    'created_at': result.created_at,
                }

        # College averages
        college_averages = CollegeEvaluationResult.objects.all()
        response_data['college_evaluation'] = [
            {
                'college': c.college,
                'average_score': c.average_score,
                'rank': c.rank
            }
            for c in college_averages
        ]

        # College comparisons
        response_data['college_comparisons'] = [
            {
                'group1': cc.group1,
                'group2': cc.group2,
                'p_value': cc.p_value,
                'reject_null': cc.reject_null,
            }
            for cc in CollegeComparisonResult.objects.all()
        ]

        # Statsmodels
        stats_result = StatsmodelsRegressionResult.objects.last()
        if stats_result:
            response_data['statsmodels'] = {
                'r2_score': stats_result.r2_score,
                'adjusted_r2': stats_result.adjusted_r2,
                'mse': stats_result.mse,
                'coefficients': stats_result.coefficients,
                'p_values': stats_result.p_values,
                'notes': stats_result.notes,
                'samples': [
                    {'index': s.index, 'actual': s.actual, 'predicted': s.predicted}
                    for s in stats_result.samples.all()
                ]
            }

        # Sklearn
        sklearn_result = SklearnRegressionResult.objects.last()
        if sklearn_result:
            response_data['sklearn'] = {
                'r2_score': sklearn_result.r2_score,
                'mse': sklearn_result.mse,
                'coefficients': sklearn_result.coefficients,
                'notes': sklearn_result.notes,
                'samples': [
                    {'index': s.index, 'actual': s.actual, 'predicted': s.predicted}
                    for s in sklearn_result.samples.all()
                ]
            }

        return Response(response_data)