import numpy as np

class LinearRegression:
    """
    A custom Multiple Linear Regression model implemented from scratch.
    Uses Ordinary Least Squares (OLS) with Ridge Regularization (L2 penalty) for stability.
    Includes built-in z-score standardization of features.
    """
    def __init__(self, alpha=1e-5):
        self.alpha = alpha  # L2 regularization term to prevent singularity
        self.coefficients = None
        self.intercept = None
        self.means = None
        self.stds = None

    def fit(self, X, y):
        """Fit the linear regression model using closed-form OLS solution."""
        # 1. Calculate statistics for scaling
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        
        # Avoid division by zero for constant features
        self.stds[self.stds == 0] = 1.0
        
        # 2. Standardize X
        X_scaled = (X - self.means) / self.stds
        
        # 3. Add column of 1s for the intercept
        n_samples = X_scaled.shape[0]
        X_design = np.hstack([np.ones((n_samples, 1)), X_scaled])
        
        # 4. Closed form solution: beta = (X_design^T * X_design + alpha * I)^-1 * X_design^T * y
        n_features = X_design.shape[1]
        I = np.eye(n_features)
        # Do not regularize the intercept term (first column)
        I[0, 0] = 0.0
        
        # Compute coefficients
        A = X_design.T @ X_design + self.alpha * I
        b = X_design.T @ y
        
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if solve fails
            beta = np.linalg.pinv(A) @ b
            
        self.intercept = float(beta[0])
        self.coefficients = beta[1:].tolist()

    def predict(self, X):
        """Predict target variables for input samples."""
        X_scaled = (X - self.means) / self.stds
        return np.dot(X_scaled, np.array(self.coefficients)) + self.intercept

    def to_dict(self):
        """Export the model parameters to a dictionary for JSON serialization."""
        return {
            "coefficients": [float(c) for c in self.coefficients],
            "intercept": float(self.intercept),
            "means": [float(m) for m in self.means],
            "stds": [float(s) for s in self.stds]
        }
