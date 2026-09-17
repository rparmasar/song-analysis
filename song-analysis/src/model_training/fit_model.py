from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, PoissonRegressor
from sklearn.model_selection import GridSearchCV

# training constants
LINEAR_MODELS_GRID_SEARCH_PARAMS = {
    "alpha": np.linspace(start=0.1, stop=1, num=20),
    "l1_ratio": np.linspace(start=0.1, stop=1, num=20),
}

RANDOM_FOREST_GRID_SEARCH_PARAMS = {
    "n_estimators": range(100, 550, 50),
    "max_depth": range(4, 8),
}

HIST_GBM_GRID_SEARCH_PARAMS = {
    "max_iter": range(100, 700, 100),
    "max_features": np.linspace(0.4, 1.0, 6),
}


def fit_elastic_net(
    train_features: pd.DataFrame,
    train_response: pd.DataFrame,
    features_to_train_on: list[str] | None = None,
    grid_search_params: dict[str, any] = LINEAR_MODELS_GRID_SEARCH_PARAMS,
    print_model_fitting_logs: bool = False,
) -> tuple[ElasticNet, float]:
    """
    fits an elastic net (normal linear regression with penalty params) with RMSE as objective and returns the best model using 5-fold CV
    """
    print("[fit_elastic_net] starting fitting procedure")
    # init grid search
    elastic_net_grid_search = GridSearchCV(
        estimator=ElasticNet(),
        param_grid=grid_search_params,
        scoring="neg_root_mean_squared_error",
        cv=5,
        verbose=2 if print_model_fitting_logs else 0,
    )
    start_time = perf_counter()
    # fit Models
    elastic_net_grid_search.fit(
        X=train_features[features_to_train_on]
        if features_to_train_on
        else train_features,
        y=train_response,
    )
    end_time = perf_counter()

    train_time = end_time - start_time

    # log best results
    print(f"[fit_elastic_net] total train time: {round(train_time, ndigits=3)} seconds")

    print(f"[fit_elastic_net] best parameters: {elastic_net_grid_search.best_params_}")
    print(f"[fit_elastic_net] best rmse: {-elastic_net_grid_search.best_score_}")

    # return best model
    return elastic_net_grid_search.best_estimator_, train_time


def fit_poisson_glm(
    train_features: pd.DataFrame,
    train_response: pd.DataFrame,
    features_to_train_on: list[str] | None = None,
    grid_search_params: dict[str, any] | None = None,
    print_model_fitting_logs: bool = False,
) -> tuple[PoissonRegressor, float]:
    """
    fits a poisson glm with RMSE as objective and returns the best model using 5-fold CV
    """
    print("[fit_poisson_glm] starting fitting procedure")
    if grid_search_params is None:
        grid_search_params = {
            "alpha": LINEAR_MODELS_GRID_SEARCH_PARAMS["alpha"],
        }

    pglm_grid_search = GridSearchCV(
        estimator=PoissonRegressor(),
        param_grid=grid_search_params,
        scoring="neg_mean_squared_log_error",
        cv=5,
        verbose=2 if print_model_fitting_logs else 0,
    )

    start_time = perf_counter()
    pglm_grid_search.fit(
        X=train_features[features_to_train_on]
        if features_to_train_on
        else train_features,
        y=train_response["popularity"],
    )
    end_time = perf_counter()
    train_time = end_time - start_time

    print(f"[fit_poisson_glm] total train time: {round(train_time, ndigits=3)} seconds")

    print(f"[fit_poisson_glm] best parameters: {pglm_grid_search.best_params_}")
    print(f"[fit_poisson_glm] best rmse: {-pglm_grid_search.best_score_}")

    return pglm_grid_search.best_estimator_, train_time


def fit_random_forest(
    train_features: pd.DataFrame,
    train_response: pd.DataFrame,
    features_to_train_on: list[str] | None = None,
    grid_search_params: dict[str, any] = RANDOM_FOREST_GRID_SEARCH_PARAMS,
    print_model_fitting_logs: bool = False,
) -> tuple[RandomForestRegressor, float]:
    """
    fits a random forest (with poisson criteria) and returns the best model using 5-fold cv
    """
    print("[fit_random_forest] starting fitting procedure")

    rf_regressor_grid_search = GridSearchCV(
        estimator=RandomForestRegressor(criterion="poisson"),
        param_grid=grid_search_params,
        scoring="neg_root_mean_squared_error",
        cv=5,
        verbose=2 if print_model_fitting_logs else 0,
    )
    start_time = perf_counter()
    rf_regressor_grid_search.fit(
        X=train_features[features_to_train_on]
        if features_to_train_on
        else train_features,
        y=train_response["popularity"],
    )
    end_time = perf_counter()

    train_time = end_time - start_time

    print(
        f"[fit_random_forest] total train time: {round(train_time, ndigits=3)} seconds"
    )

    print(
        f"[fit_random_forest] best parameters: {rf_regressor_grid_search.best_params_}"
    )
    print(f"[fit_random_forest] best rmse: {-rf_regressor_grid_search.best_score_}")

    return rf_regressor_grid_search.best_estimator_, train_time


def fit_hist_gbm(
    train_features: pd.DataFrame,
    train_response: pd.DataFrame,
    features_to_train_on: list[str] | None = None,
    grid_search_params: dict[str, any] = HIST_GBM_GRID_SEARCH_PARAMS,
    print_model_fitting_logs: bool = False,
) -> tuple[HistGradientBoostingRegressor, float]:
    """
    fits a histogram-based gradient boosting model (with poisson criteria) and returns the best model using 5-fold cv.

    this is scikit-learn's take on LightGBM.
    """
    print("[fit_hist_gbm] starting fitting procedure")

    hgbr_grid_search = GridSearchCV(
        estimator=HistGradientBoostingRegressor(loss="poisson"),
        param_grid=grid_search_params,
        scoring="neg_root_mean_squared_error",
        cv=5,
        verbose=2 if print_model_fitting_logs else 0,
    )
    start_time = perf_counter()
    hgbr_grid_search.fit(
        X=train_features[features_to_train_on]
        if features_to_train_on
        else train_features,
        y=train_response["popularity"],
    )
    end_time = perf_counter()

    train_time = end_time - start_time

    print(f"[fit_hist_gbm] total train time: {round(train_time, ndigits=3)} seconds")

    print(f"[fit_hist_gbm] best parameters: {hgbr_grid_search.best_params_}")
    print(f"[fit_hist_gbm] best rmse: {-hgbr_grid_search.best_score_}")

    return hgbr_grid_search.best_estimator_, train_time
