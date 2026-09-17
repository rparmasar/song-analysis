from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, PoissonRegressor
from sklearn.model_selection import GridSearchCV

# training constants
LINEAR_MODELS_GRID_SEARCH_PARAMS = {
    "alpha": np.linspace(start=0.1, stop=1, num=20),
    "l1_ratio": np.linspace(start=0.1, stop=1, num=20),
}


def fit_elastic_net(
    train_features: pd.DataFrame,
    train_response: pd.DataFrame,
    grid_search_params: dict[str, any] = LINEAR_MODELS_GRID_SEARCH_PARAMS,
    print_model_fitting_logs: bool = False,
) -> ElasticNet:
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
    elastic_net_grid_search.fit(X=train_features, y=train_response)
    end_time = perf_counter()
    # log best results
    print(
        f"[fit_elastic_net] total train time: {round(end_time - start_time, ndigits=3)} seconds"
    )

    print(f"[fit_elastic_net] best parameters: {elastic_net_grid_search.best_params_}")
    print(f"[fit_elastic_net] best rmse: {-elastic_net_grid_search.best_score_}")

    # return best model
    return elastic_net_grid_search.best_estimator_


def fit_poisson_glm(
    train_features: pd.DataFrame,
    train_response: pd.DataFrame,
    grid_search_params: dict[str, any] | None = None,
    print_model_fitting_logs: bool = False,
) -> PoissonRegressor:
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
    pglm_grid_search.fit(X=train_features, y=train_response["popularity"])
    end_time = perf_counter()
    print(
        f"[fit_poisson_glm] total train time: {round(end_time - start_time, ndigits=3)} seconds"
    )

    print(f"[fit_poisson_glm] best parameters: {pglm_grid_search.best_params_}")
    print(f"[fit_poisson_glm] best rmse: {-pglm_grid_search.best_score_}")

    return pglm_grid_search.best_estimator_


def fit_random_forest(train_features: pd.DataFrame, train_response: pd.DataFrame):
    pass


def fit_light_gbm(train_features: pd.DataFrame, train_response: pd.DataFrame):
    pass
