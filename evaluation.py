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


def summarize_evaluation(metrics, train_score, test_score):
    """Return a short plain-English summary of model quality."""
    summary = ["Evaluation summary:"]

    if metrics["r2"] >= 0.7:
        summary.append("- R2 is strong for this test split.")
    elif metrics["r2"] >= 0.4:
        summary.append("- R2 is moderate, so the model captures some signal.")
    else:
        summary.append("- R2 is low, so the model may need more tuning.")

    if metrics["rmse"] <= 0.75:
        summary.append("- RMSE is under one quality point, which is easy to interpret.")
    else:
        summary.append("- RMSE is above one quality point, so errors may be noticeable.")

    score_gap = train_score - test_score
    if score_gap <= 15:
        summary.append("- Train and test scores are close, with no large overfitting signal.")
    else:
        summary.append("- Train score is much higher than test score; check for overfitting.")

    return "\n".join(summary) + "\n"
