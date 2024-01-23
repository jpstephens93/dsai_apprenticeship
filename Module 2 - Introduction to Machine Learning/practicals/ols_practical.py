from statsmodels.formula.api import ols
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# Generate data with 2 features and a target variable
def generate_data_3d(n):
    np.random.seed(n)
    X = np.stack([np.random.normal(scale=2, size=n), np.random.normal(scale=5, size=n)]).T
    beta = np.absolute(np.random.randn(3))
    # add slight some standard normal noise
    y = beta[0] + beta[1] * X[:, 0] + beta[2] * X[:, 1] + np.random.normal(scale=1, size=n)
    return X, y, beta


# Using OLS - Lesson
np.random.seed(5)

# Generate monotonic increasing values for x, 1 to 100
x = np.linspace(1, 100, 100)

# Add a little random noise to each x, to get a y value
y = [i + 50 + np.random.normal(loc=0, scale=10) for i in x]

sns.scatterplot(x=x, y=y)
plt.show()

# Reshape X
X = x.reshape(-1, 1)
# Add a column of ones
X = np.hstack((np.ones(X.shape), X))

# Calculate the optimal parameters with the closed form solution equation
beta = np.linalg.inv(X.T @ X) @ np.dot(X.T, y)

# Calculate the prediction for each observation
Y_Jred = X @ beta

# Plot the predictions
plt.scatter(x=x, y=y)
plt.plot([min(x), max(x)], [min(Y_Jred), max(Y_Jred)], color='red')  # linear regression line in red
plt.show()


# Calculate the MSE for this line of best fit
def mse(y, y_Jred):
    return np.mean((y - y_Jred)**2)


print(f'MSE for OLS: {mse(y, Y_Jred):.3f}')

# Multivariate OLS
n = 100
x, y, beta_true = generate_data_3d(n)
X = np.hstack((np.ones((100, 1)), x))

# Calculate the optimal parameters with the closed form solution for OLS
beta = np.linalg.inv(X.T @ X) @ np.dot(X.T, y)
print(beta)

# Calculate the prediction for each obsevation
Y_pred_ols = X @ beta

# Practical
# 1) Generate Data

# Define the number of samples and explanatory variables
num_samples = 100
num_features = 7

# Generate random values for the explanatory variables
np.random.seed(0)
X = np.hstack((np.ones((num_samples, 1)), np.random.normal(size=(num_samples, num_features))))

# Define the true parameter values for the linear regression
true_coefs = np.array([0, 2, -3, 4,-2, -10, 5, 9])

# Generate the target values using the explanatory variables
y = X @ true_coefs + np.random.normal(size=num_samples)

# Combine the explanatory variables and response values into a single data frame
df = pd.DataFrame(np.hstack([X, y.reshape(-1, 1)]), columns=["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "y"])

# 2) Visualise Data
sns.pairplot(df, x_vars=["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"], y_vars=['y'])
plt.show()

# 3) Perform OLS Regression
model = ols(formula="y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7", data=df)
results = model.fit()

print(results.summary())

# 4) Interpret the Weights
print("The parameters are the weights assigned to each x feature. So they behave like a gradient to y (change by 1"
      "results in a change of w all else constant.")

# 5) Determine Significance
print("Those parameter weights with p-values less than 0.05 fulfill the alternative hypothesis, whereas the bias term"
      "does not.")

# 6) Determine overall significance
results.f_test(np.identity(len(results.params)))
print("Overall model has p-value less than 0.05, hence the Null Hypothesis can be rejected.")

# 7) Interpret R^2
print('R squared score which is 0.996. This means that 99.6% of the variation on the target variable is explained by '
      'the mutivariate linear regression model.')

# 8) Generate Predictions
pred = results.predict()
