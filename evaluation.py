import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(y_true, y_pred):
    """Return common regression metrics for a model report."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def format_regression_metrics(metrics):
    """Format regression metrics for printing or writing to a text report."""
    return (
        "Mean absolute error: %.3f\n"
        "Root mean squared error: %.3f\n"
        "R2 score: %.3f\n"
    ) % (
        metrics["mae"],
        metrics["rmse"],
        metrics["r2"],
    )
