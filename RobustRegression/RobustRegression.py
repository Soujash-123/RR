import numpy as np
from scipy.optimize import minimize

class RobustRegressionWithUncertainty:
    def __init__(self, alpha=1.0, lambda_1=0.01, lambda_2=0.01):
        self.alpha = alpha  # Regularization parameter for predictions
        self.lambda_1 = la mbda_1  # Regularization parameter for beta
        self.lambda_2 = lambda_2  # Regularization parameter for gamma
        self.beta = None  # Parameters for predictions
        self.gamma = None  # Parameters for uncertainty
        self.n_features = None
    
    def fit(self, X, y):
        self.n_features = X.shape[1]
        # Initialize parameters
        self.beta = np.random.rand(self.n_features)
        self.gamma = np.random.rand(self.n_features)

        # Minimize the loss function
        result = minimize(self.loss_function, 
                          x0=np.concatenate((self.beta, self.gamma)), 
                          args=(X, y), 
                          method='L-BFGS-B')
        
        # Extract optimized parameters
        self.beta = result.x[:self.n_features]
        self.gamma = result.x[self.n_features:]
    
    def predict(self, X):
        """ Predict the expected values and uncertainty estimates. """
        mean_prediction = X @ self.beta
        variance_estimate = self.estimate_variance(X)
        return mean_prediction, variance_estimate
    
    def estimate_variance(self, X):
        """ Estimate the variance using the uncertainty function. """
        # Here, a simple linear model for variance is used; you can change this
        return np.exp(X @ self.gamma)  # Exponentiated to ensure positive variance
    
    def loss_function(self, params, X, y):
        """ Compute the total loss function. """
        beta = params[:self.n_features]
        gamma = params[self.n_features:]
        
        predictions = X @ beta
        variance_estimates = np.exp(X @ gamma)  # Exponentiated for positivity
        
        # Calculate loss
        residuals = y - predictions
        log_likelihood = -0.5 * np.sum((residuals**2 / variance_estimates) + np.log(variance_estimates))
        regularization = (self.lambda_1 * np.sum(beta**2)) + (self.lambda_2 * np.sum(gamma**2))
        
        return -log_likelihood + regularization

# Example usage
if __name__ == "__main__":
    # Sample data
    np.random.seed(0)
    X = np.random.rand(100, 3)
    true_beta = np.array([1.5, -2.0, 0.5])
    y = X @ true_beta + np.random.normal(0, 0.5, size=100)  # Adding noise

    # Initialize and fit the model
    model = RobustRegressionWithUncertainty()
    model.fit(X, y)

    # Predict
    predictions, uncertainties = model.predict(X)
    print("Predictions:", predictions)
    print("Uncertainties:", uncertainties)

