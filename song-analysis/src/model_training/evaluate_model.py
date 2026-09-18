# houses functions to compute metrics to evaluate models
import pandas as pd
from matplotlib.axes import Axes
from sklearn.metrics import r2_score, root_mean_squared_error

from src.visualization.predictions import plot_train_test_dist


def compute_model_performance(
    train_predictions: pd.DataFrame,
    train_response: pd.DataFrame,
    test_predictions: pd.DataFrame,
    test_response: pd.DataFrame,
    train_time: float,
    figsize: tuple[int, int] | None = (15, 6),
) -> tuple[dict, list[Axes]]:
    """
    computes r^2 and rmse for both train and test set and also visualizes the predictions against the ground truth.
    """
    # compute metrics
    train_r2 = r2_score(train_response, train_predictions)
    train_rmse = root_mean_squared_error(train_response, train_predictions)
    test_r2 = r2_score(test_response, test_predictions)
    test_rmse = root_mean_squared_error(test_response, test_predictions)

    metrics_dict = {
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "training_time_seconds": train_time,
    }

    # compute plot
    axes = plot_train_test_dist(
        train_predictions=train_predictions,
        train_response=train_response,
        test_response=test_response,
        test_predictions=test_predictions,
        figsize=figsize,
    )

    return metrics_dict, axes


# TODO: this will be used when scoring the dancehall tracks
def compute_shap_values():
    pass
