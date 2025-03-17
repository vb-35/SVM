import numpy as np
import matplotlib.pyplot as plt

class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    def _init_weights_bias(self, X):
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0
    
    def _get_classlabels(self, y):
        return np.where(y <= 0, -1, 1)
    
    def _satisfy_constraint(self, x, y):
        return y * (np.dot(x, self.w) + self.b) >= 1
    
    def fit(self, X, y):
        y = self._get_classlabels(y)
        self._init_weights_bias(X)
        
        # Gradient descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # If sample satisfies constraint, only regularization gradient
                if self._satisfy_constraint(x_i, y[idx]):
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                # If constraint not satisfied, apply loss gradient and regularization
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y[idx] * x_i)
                    self.b -= self.lr * (-y[idx])
        
        return self
    
    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)
    
    def score(self, X, y):
        y_true = self._get_classlabels(y)
        y_pred = self.predict(X)
        return np.mean(y_true == y_pred)
    
    def plot_decision_boundary(self, X, y):
        if X.shape[1] != 2:
            raise ValueError("This function only works for 2D data")
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        y = self._get_classlabels(y)
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        #Decision boundary
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        
        Z = np.dot(np.c_[xx.ravel(), yy.ravel()], self.w) + self.b
        Z = Z.reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[0], colors='k', linestyles=['-'], linewidths=2)
        ax.contour(xx, yy, Z, levels=[-1, 1], colors='grey', linestyles=['--', '--'], linewidths=1)
        
        # Plot data points
        ax.scatter(X[y==1, 0], X[y==1, 1], label="Class 1", marker='+', s=120)
        ax.scatter(X[y==-1, 0], X[y==-1, 1], label="Class -1", marker='o', s=120)
        
        
        plt.title("SVM Decision Boundary")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.grid(True)
        plt.show()