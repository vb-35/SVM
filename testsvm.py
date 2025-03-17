from SVM import LinearSVM
import numpy as np

#sample data
np.random.seed(42)
X = np.random.randn(100, 2)
y = np.array([1 if x[0] + x[1] > 0 else -1 for x in X])

X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

svm = LinearSVM(learning_rate=0.01, lambda_param=0.01, n_iters=1000)
svm.fit(X_train, y_train)



predictions = svm.predict(X_test)
accuracy = svm.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
svm.plot_decision_boundary(X, y)