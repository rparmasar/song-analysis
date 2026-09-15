# Contain functions for fitting and diagnosing models
from pandas import DataFrame
from matplotlib.axes import Axes
from matplotlib.pyplot import subplots
from seaborn import histplot

from typing import Optional

def plot_train_test_dist(train_predictions: DataFrame, train_response: DataFrame, test_predictions: DataFrame, test_response: DataFrame, figsize: "Optional[tuple[int, int]]" = (15, 6)) -> "list[Axes]":
    """
    Uses the input dataframes to configure and return 2 `Axes` that can be used for plotting with `matplotlib.pyplot.show()`
    """
    # Create a figure with 2 sub-plots
    _, axes = subplots(nrows=1, ncols=2, figsize = figsize)

    # Configure Train Set plots
    histplot(
        train_response["popularity"],
        color="grey",
        label = "Observed Distribution",
        ax= axes[0]
    )
    histplot(
        train_predictions["expected_popularity"],
        color="blue",
        element="step",
        label = "Predicted Distribution",
        ax=axes[0]
    )
    
    # Configure Test Set plots
    histplot(
        test_response["popularity"],
        color="grey",
        label = "Observed Distribution",
        ax= axes[1]
    )
    histplot(
        test_predictions["expected_popularity"],
        color="blue",
        element="step",
        label = "Predicted Distribution",
        ax=axes[1]
    )

    # Add titles + legends to plots
    axes[0].set_xlabel("popularity")
    axes[0].set_title("Train Set Popularity Distribution")
    axes[0].legend(loc='upper left')

    axes[1].set_xlabel("popularity")
    axes[1].set_title("Test Set Popularity Distribution")
    axes[1].legend(loc='upper left')

    return axes