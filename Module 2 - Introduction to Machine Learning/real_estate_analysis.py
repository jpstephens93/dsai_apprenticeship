import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('data/real_estate_valuation_data_set.csv')
df.head()

# Q1.
df.info()
values_missing = df.isna().any().any()

# Q2.
df_new = df.copy()
df_new.drop(columns=['X1 transaction date'], inplace=True)

# Q3.
hist_y = df_new['Y house price of unit area'].hist()
plt.show()

# Q4.
fig, ax = plt.subplots(figsize=(10, 25), dpi=50)
cols = ['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude',
        'X6 longitude']
hist_per_input = df_new[cols].hist(ax=ax, layout=(5, 1), column=cols)
plt.show()

# Q5.
house_price_mean = df_new['Y house price of unit area'].mean()
house_price_median = df_new['Y house price of unit area'].median()

# Q6.
df_new_scaled = (df_new.iloc[:, :-1] - df_new.iloc[:, :-1].min()) / (
        df_new.iloc[:, :-1].max() - df_new.iloc[:, :-1].min())
df_new_scaled['Y house price of unit area'] = df_new['Y house price of unit area']

# Q7.
X1 = df_new_scaled[[
        'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude',
        'X6 longitude'
]]
y1 = df_new_scaled['Y house price of unit area']
reg1 = LinearRegression().fit(X1, y1)

# Q8.
reg1_R2 = reg1.score(X1, y1)

# Q9.
coefficients = reg1.coef_.tolist()

# Q10.
X2 = df_new_scaled[['X2 house age', 'X5 latitude', 'X6 longitude']]
y2 = df_new_scaled['Y house price of unit area']
reg2 = LinearRegression().fit(X2, y2)

# Q11.
reg2_R2 = reg2.score(X2, y2)

# Q12.
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.25, random_state=101)

# Q13.
reg3 = LinearRegression().fit(X_train, y_train)

# Q14.
y_pred = reg3.predict(X_test).tolist()
pickled = pickle.dumps(y_pred, 0)
pickled

# Q15.
mean_squared_error(y_pred, y_test)
reg3.score(X_test, y_test)
