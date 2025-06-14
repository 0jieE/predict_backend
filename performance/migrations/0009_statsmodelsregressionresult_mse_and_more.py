# Generated by Django 5.2.1 on 2025-06-04 01:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('performance', '0008_sklearnregressionresult_statsmodelsregressionresult_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='statsmodelsregressionresult',
            name='mse',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='statsmodelsregressionresult',
            name='p_values',
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='statsmodelsregressionresult',
            name='r2_score',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='sklearnpredictionsample',
            name='actual',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='sklearnpredictionsample',
            name='index',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='sklearnpredictionsample',
            name='predicted',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='sklearnregressionresult',
            name='coefficients',
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='statsmodelspredictionsample',
            name='actual',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='statsmodelspredictionsample',
            name='index',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='statsmodelspredictionsample',
            name='predicted',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='statsmodelsregressionresult',
            name='coefficients',
            field=models.JSONField(blank=True, null=True),
        ),
    ]
