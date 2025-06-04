# serializers.py
from rest_framework import serializers
from .models import FacultyMember, FacultyPerformance

class FacultyMemberSerializer(serializers.ModelSerializer):
    class Meta:
        model = FacultyMember
        fields = ['id', 'name', 'position', 'sex', 'campus', 'college']

class FacultyPerformanceSerializer(serializers.ModelSerializer):
    class Meta:
        model = FacultyPerformance
        fields = ['id', 'faculty', 'school_year', 'semester', 'student_evaluation', 'workload_units', 'num_preparations', 'deloading']

