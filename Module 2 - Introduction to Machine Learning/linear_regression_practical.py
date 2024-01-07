import numpy as np


# Generate data with 2 features and a target variable
def generate_data_3d(n):
    np.random.seed(n)
    X = np.stack([np.random.normal(scale=2, size=n), np.random.normal(scale=5, size=n)]).T
    beta = np.absolute(np.random.randn(3))
    # add slight some standard normal noise
    y = beta[0] + beta[1] * X[:, 0] + beta[2] * X[:, 1] + np.random.normal(scale=1, size=n)
    return X, y, beta


n = 100
X, y, beta_true = generate_data_3d(n)

# Extract each feature as separate vectors
x1 = X[:, 0]
x2 = X[:, 1]

# Initial values for the weight and bias
b = 0
w1 = 0
w2 = 0

eta = 0.001  # The learning rate
iterations = 1000  # The number of iterations to perform gradient descent

n = float(len(x1))  # Number of observations in X

# Performing Gradient Descent
for i in range(iterations):
    # write your code here
    y_pred = b + w1 * x1 + w2 * x2  # predicted value of y
    D_w1 = (-2 / n) * sum(x1 * (y - y_pred))  # derivative of loss function (MSE) wrt w1
    D_w2 = (-2 / n) * sum(x2 * (y - y_pred))  # derivative of loss function (MSE) wrt w2
    D_b = (-2 / n) * sum(y - y_pred)  # derivative of loss function (MSE) wrt b
    w1 = w1 - eta * D_w1  # updated value for w1
    w2 = w2 - eta * D_w2  # updated value for w2
    b = b - eta * D_b  # updated value for b

print(f'Estimated parameters: bias: {b:.2f}, weight 1: {w1:.2f}, weight 2: {w2:.2f}')

# Add a dim to x with ones to account for the bias term
Xmat = np.hstack((np.ones([X.shape[0], 1]), X))

# Initial values for the weights (which include the bias term or beta_0)
beta = np.zeros(Xmat.shape[1])
eta = 0.001  # The learning rate
iterations = 1000  # The number of iterations to perform gradient descent

n = float(Xmat.shape[0])  # Number of observations in X

# Performing Gradient Descent
for i in range(iterations):
    # i = 1
    y_pred = Xmat @ beta  # using all 3 terms in beta vector
    D_beta = -(2 / n) * Xmat.T @ (y - y_pred)  # take derivative of beta vector values
    beta = beta - eta * D_beta  # update the beta vector

print(f'beta: {beta}')
np.isclose(beta, [b, w1, w2])

# Using OLS
