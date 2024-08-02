from sklearn.model_selection import GridSearchCV
from custom_logit import CustomLogit
import logging

def perform_grid_search(X_train, y_train, use_regularization=True):
    param_grid = {'alpha': [0.01, 0.1, 1, 10, 100, 1000]}

    grid_search = GridSearchCV(estimator=CustomLogit(use_regularization=use_regularization), param_grid=param_grid, cv=5, scoring='f1', verbose=True, n_jobs=-1)

    try:
        grid_search.fit(X_train, y_train)
        logging.info(f"Best Alpha: {grid_search.best_params_}")
        logging.info(f"Best Score: {grid_search.best_score_}")
    except Exception as e:
        logging.error(f"An error occurred during grid search: {e}")
        raise

    return grid_search.best_params_, grid_search.best_score_

# Example usage (you need to define X_train and y_train before this point)
# X_train = ...
# y_train = ...
# best_params, best_score = perform_grid_search(X_train, y_train, use_regularization=True)
# print(f"Best Params: {best_params}")
# print(f"Best Score: {best_score}")
