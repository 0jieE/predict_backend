from django.core.management.base import BaseCommand
from performance.models import FacultyPerformance

class Command(BaseCommand):
    help = "Set NULL values for workload_units and num_preparations to 0"

    def handle(self, *args, **kwargs):
        updated_workload = FacultyPerformance.objects.filter(workload_units__isnull=True).update(workload_units=0)
        updated_preparations = FacultyPerformance.objects.filter(num_preparations__isnull=True).update(num_preparations=0)

        self.stdout.write(self.style.SUCCESS(
            f"Updated {updated_workload} records with NULL workload_units, and {updated_preparations} records with NULL num_preparations."
        ))
