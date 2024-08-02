# GRIDSEARCHCV-LOGIT-STATSMODELS

This project demonstrates how to perform grid search with `statsmodels` Logit models using Scikit-learn's `GridSearchCV`.

## Features

- Customizable logistic regression model with an option to use regularization.
- Grid search functionality to find the optimal hyperparameters for the model.
- Logging for easy debugging and tracking.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/TusharNair04/gridsearchcv-logit-statsmodels.git
   cd GRIDSEARCHCV-LOGIT-STATSMODELS
   ```

2. Install the necessary dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Modify `run_grid_search.py` to include your training data.
2. Run the grid search script to find the best hyperparameters.

## Example

```python
# Define your training data
X_train = ...
y_train = ...

# Perform grid search
best_params, best_score = perform_grid_search(X_train, y_train, use_regularization=True)
print(f"Best Params: {best_params}")
print(f"Best Score: {best_score}")
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.
