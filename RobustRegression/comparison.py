import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import minimize
from RobustRegression import *

def create_synthetic_data(n_samples=100):
    np.random.seed(0)
    X = np.random.rand(n_samples, 3)
    true_beta = np.array([1.5, -2.0, 0.5])
    y = X @ true_beta + np.random.normal(0, 0.5, size=n_samples)  # Adding noise
    return X, y

def plot_results(X, y, model_predictions, uncertainties, model_name):
    plt.scatter(y, model_predictions, label=model_name)
    plt.fill_between(y, model_predictions - uncertainties, model_predictions + uncertainties, alpha=0.2)
    plt.plot([min(y), max(y)], [min(y), max(y)], 'k--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'{model_name} Predictions vs True Values')
    plt.legend()
    plt.show()


# Generate synthetic data
X, y = create_synthetic_data()

# Your model
robust_model = RobustRegressionWithUncertainty()
robust_model.fit(X, y)
robust_predictions, robust_uncertainties = robust_model.predict(X)

# Bayesian Linear Regression
bayesian_model = BayesianRidge()
bayesian_model.fit(X, y)
bayesian_predictions = bayesian_model.predict(X)

# Compute prediction uncertainties for Bayesian Ridge
bayesian_uncertainties = np.array([
    np.sqrt(np.dot(X[i], np.dot(bayesian_model.sigma_, X[i].T))) for i in range(X.shape[0])
])

# Gaussian Process Regression
kernel = RBF(length_scale=1.0)
gp_model = GaussianProcessRegressor(kernel=kernel, alpha=0.5)
gp_model.fit(X, y)
gp_predictions, gp_uncertainties = gp_model.predict(X, return_std=True)

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X, y)
ridge_predictions = ridge_model.predict(X)
ridge_uncertainties = np.full_like(y, 0.5)  # Assume constant uncertainty for Ridge

# Plotting results for comparison
plot_results(X, y, robust_predictions, robust_uncertainties, "Robust Regression")
plot_results(X, y, bayesian_predictions, bayesian_uncertainties, "Bayesian Regression")
plot_results(X, y, gp_predictions, gp_uncertainties, "Gaussian Process Regression")
plot_results(X, y, ridge_predictions, ridge_uncertainties, "Ridge Regression")

# Optionally, calculate metrics like MSE or R-squared for each model
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

# Evaluate models
models = {
    "Robust Regression": (robust_predictions, robust_uncertainties),
    "Bayesian Regression": (bayesian_predictions, bayesian_uncertainties),
    "Gaussian Process Regression": (gp_predictions, gp_uncertainties),
    "Ridge Regression": (ridge_predictions, ridge_uncertainties),
}

for model_name, (preds, _) in models.items():
    mse, r2 = evaluate_model(y, preds)
    print(f"{model_name} -> MSE: {mse:.4f}, R^2: {r2:.4f}")

