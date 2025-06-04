from django.core.management.base import BaseCommand
from performance.ml.stats_regression import run_stats_regression
from performance.ml.sklearn_regression import run_sklearn_regression
import numpy as np


class Command(BaseCommand):
    help = 'Compare statsmodels vs sklearn regression results on student evaluation prediction.'

    def handle(self, *args, **kwargs):
        self.stdout.write(self.style.MIGRATE_HEADING("Running statsmodels regression..."))
        stats_result, stats_preds = run_stats_regression()

        self.stdout.write(self.style.MIGRATE_HEADING("\nRunning sklearn regression..."))
        sklearn_result = run_sklearn_regression()

        self.stdout.write(self.style.SUCCESS("\n=== Comparison Results ==="))

        self.stdout.write("\n[Statsmodels]")
        self.stdout.write(f"Adjusted R^2: {stats_result.rsquared_adj:.4f}")
        self.stdout.write(f"P-values:\n{stats_result.pvalues}")

        self.stdout.write("\n[Scikit-learn]")
        self.stdout.write(f"R^2 Score: {sklearn_result['r2_score']:.4f}")
        self.stdout.write(f"MSE: {sklearn_result['mse']:.4f}")

        if stats_result.rsquared_adj > sklearn_result['r2_score']:
            self.stdout.write(self.style.NOTICE("\nStatsmodels performed slightly better in terms of adjusted R^2."))
        else:
            self.stdout.write(self.style.NOTICE("\nScikit-learn performed slightly better in terms of prediction (R^2)."))

        # Show a few example predictions vs actual values
        self.stdout.write("\n[Sample Predictions Comparison]")
        n_samples = min(5, len(stats_preds))  # show up to 5 samples

        self.stdout.write(f"{'Index':<6}{'Actual':>10}{'Statsmodels Pred':>20}{'Sklearn Pred':>20}")
        for i in range(n_samples):
            actual = sklearn_result['true'].iloc[i] if hasattr(sklearn_result['true'], 'iloc') else sklearn_result['true'][i]
            stat_pred = stats_preds[i]
            skl_pred = sklearn_result['predictions'][i]
            self.stdout.write(f"{i:<6}{actual:10.4f}{stat_pred:20.4f}{skl_pred:20.4f}")

        # Interpret coefficients
        self.stdout.write(self.style.SUCCESS("\n[Statsmodels Coefficients Interpretation]"))
        summary = stats_result.summary2().tables[1]

        for feature in summary.index:
            if feature == 'const':
                continue
            coef = summary.loc[feature, 'Coef.']
            pval = summary.loc[feature, 'P>|t|']

            if coef > 0:
                direction = "increase"
            else:
                direction = "decrease"

            self.stdout.write(
                f"A unit increase in '{feature}' is associated with a {abs(coef):.4f} {direction} in student evaluations "
                f"(p-value: {pval:.4f})."
            )

        # Optionally, highlight statistically significant predictors
        sig_features = summary[summary['P>|t|'] < 0.05]
        if not sig_features.empty:
            self.stdout.write(self.style.WARNING("\n[Significant Predictors (p < 0.05)]"))
            for feature in sig_features.index:
                coef = sig_features.loc[feature, 'Coef.']
                direction = "increase" if coef > 0 else "decrease"
                self.stdout.write(f"'{feature}' significantly contributes a {abs(coef):.4f} {direction}.")
        else:
            self.stdout.write(self.style.WARNING("\nNo statistically significant predictors found (p < 0.05)."))

        # Interpret sklearn coefficients
        self.stdout.write(self.style.SUCCESS("\n[Scikit-learn Coefficients Interpretation]"))

        coef_values = sklearn_result['model'].named_steps['regressor'].coef_
        feature_names = sklearn_result['feature_names']  # You should return this in your run_sklearn_regression()

        for feature, coef in zip(feature_names, coef_values):
            direction = "increase" if coef > 0 else "decrease"
            self.stdout.write(
                f"A unit increase in '{feature}' is associated with a {abs(coef):.4f} {direction} in predicted student evaluations."
            )
