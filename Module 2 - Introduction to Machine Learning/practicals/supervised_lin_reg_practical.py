import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LinearRegression

np.random.seed(46)
n_samples = 100
X = np.linspace(0, 1, n_samples)
y = 2 * X + 0.7 * np.random.randn(n_samples)

# add some outliers to the dataset
outliers_indices = [60, 65, 70, 75, 80, 45, 50, 61]
y[outliers_indices] = y[outliers_indices] + 16

plt.scatter(x=X, y=y)
plt.show()

lr = LinearRegression()
lr.fit(X.reshape(-1, 1), y)

y_pred = lr.predict(X.reshape(-1, 1))
plt.scatter(x=X, y=y)
plt.plot(X, y_pred, color='r')
plt.show()

lasso = Lasso(alpha=1)
lasso.fit(X.reshape(-1, 1), y)

y_pred = lasso.predict(X.reshape(-1, 1))

plt.figure()
plt.scatter(X, y)
plt.plot(X, lr.predict(X.reshape(-1, 1)), color='red', label='Linear Regression')
plt.plot(X, lasso.predict(X.reshape(-1, 1)), color='green', label='Lasso Regression')
plt.legend()
plt.show()
