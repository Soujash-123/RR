import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge
import yfinance as yf
from scipy.optimize import minimize

# Your Robust Regression with Uncertainty Estimates class
class RobustRegressionWithUncertainty:
    def __init__(self, alpha=1.0, lambda_1=0.01, lambda_2=0.01):
        self.alpha = alpha
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.beta = None
        self.gamma = None
        self.n_features = None

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.beta = np.random.rand(self.n_features)
        self.gamma = np.random.rand(self.n_features)

        result = minimize(self.loss_function, 
                          x0=np.concatenate((self.beta, self.gamma)), 
                          args=(X, y), 
                          method='L-BFGS-B')

        self.beta = result.x[:self.n_features]
        self.gamma = result.x[self.n_features:]

    def predict(self, X):
        mean_prediction = X @ self.beta
        variance_estimate = self.estimate_variance(X)
        return mean_prediction, variance_estimate

    def estimate_variance(self, X):
        return np.exp(X @ self.gamma)

    def loss_function(self, params, X, y):
        beta = params[:self.n_features]
        gamma = params[self.n_features:]

        predictions = X @ beta
        variance_estimates = np.exp(X @ gamma)

        residuals = y - predictions
        log_likelihood = -0.5 * np.sum((residuals**2 / variance_estimates) + np.log(variance_estimates))
        regularization = (self.lambda_1 * np.sum(beta**2)) + (self.lambda_2 * np.sum(gamma**2))

        return -log_likelihood + regularization


# Fetch historical stock data using yfinance
stock_symbol = 'AAPL'
data = yf.download(stock_symbol, start='2022-01-01', end='2024-01-01')
data = data[['Close']]  # Using only the closing prices
data.reset_index(inplace=True)

# Prepare the dataset
data['Date'] = data['Date'].map(lambda x: x.toordinal())  # Convert dates to ordinal for regression
X = data[['Date']].values
y = data['Close'].values

# Fit the Robust Regression model
robust_model = RobustRegressionWithUncertainty()
robust_model.fit(X, y)
robust_predictions, robust_uncertainties = robust_model.predict(X)

# Plotting results
plt.figure(figsize=(10, 5))
plt.plot(data['Date'], y, label='True Prices', color='blue')
plt.plot(data['Date'], robust_predictions, label='Robust Regression Predictions', color='orange')
plt.fill_between(data['Date'], robust_predictions - robust_uncertainties, 
                 robust_predictions + robust_uncertainties, alpha=0.2, color='orange', label='Uncertainty')
plt.title(f'Robust Regression with Uncertainty Estimates for {stock_symbol}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig('robust_regression_uncertainty.png')
plt.close()

