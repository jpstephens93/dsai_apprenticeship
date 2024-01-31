from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

plt.rcParams["figure.dpi"] = 120

data = pd.read_csv("Module 2 - Introduction to Machine Learning/data/seoul_bike_data.csv")

# Q1.
categorical = [0, 4, 11, 12, 13]
continuous = [1, 2, 3, 5, 6, 7, 8, 9, 10]

# Q2.
data_date = data.copy()

date_converted = pd.to_datetime(data_date['Date'], format="%d/%m/%Y")

ref_date = datetime(2017, 1, 1)

day_count = []
for dte in date_converted:
    # dte = date_converted[0]
    day_count.append((dte - ref_date).days)

data_date['DayCount'] = day_count

# Q3.
data_season = data_date.copy()

data_season["Spring"] = data_season['Seasons'].str.contains('Spring')
data_season["Summer"] = data_season['Seasons'].str.contains('Summer')
data_season["Autumn"] = data_season['Seasons'].str.contains('Autumn')
data_season["Winter"] = data_season['Seasons'].str.contains('Winter')

sp = data_season.sample(3)

# Q4.
data_humidity = data_season.copy()

data_humidity.replace(
    {'30%-70%': sum([30, 70]) / 2, '<30%': sum([0, 30]) / 2, '>70%': sum([70, 100]) / 2}, inplace=True
)
data_humidity["Humidity(%)"].value_counts()

# Q5.
data.describe()
cont_cols = data.describe().columns.tolist()

plt.figure()
continuous_fig = data.hist(cont_cols)
plt.tight_layout()
plt.show()

# Q6.
data_binary = data_humidity.copy()

data_binary["Zero Solar Radiation"] = [True if x < 0.1 else False for x in data_binary["Solar Radiation (MJ/m2)"]]
data_binary["Zero Snowfall"] = [True if x < 0.1 else False for x in data_binary["Snowfall (cm)"]]
data_binary["Zero Rainfall"] = [True if x < 0.1 else False for x in data_binary["Rainfall(mm)"]]
data_binary["Max Visibility"] = [
    True if x > (max(data_binary["Visibility (10m)"]) - 0.1) else False for x in data_binary["Visibility (10m)"]
]

data_binary["Zero Solar Radiation"].value_counts()

# Q7.
data_z = (data[cont_cols] - data[cont_cols].mean()) / data[cont_cols].std()

normalise_fig = data_z.hist()
plt.tight_layout()
plt.show()

# Q8.
data_time_categories = data_binary.copy()

data_time_categories["Morning"] = (data_time_categories['Hour'] >= 6) & (data_time_categories['Hour'] <= 10)
data_time_categories["Afternoon"] = (data_time_categories['Hour'] >= 11) & (data_time_categories['Hour'] <= 16)
data_time_categories["Evening"] = (data_time_categories['Hour'] >= 17) & (data_time_categories['Hour'] <= 19)
data_time_categories["Night"] = (data_time_categories['Hour'] >= 20) & (data_time_categories['Hour'] <= 23)
data_time_categories["Early Morning"] = (data_time_categories['Hour'] >= 0) & (data_time_categories['Hour'] <= 5)

data_time_categories["Morning"].value_counts()

# Q9.
bike_hour_dependency = sns.violinplot(data, x='Hour', y='Rented Bike Count')
plt.show()

# Q10.
mean_count = data[['Hour', 'Rented Bike Count']].groupby('Hour').mean()['Rented Bike Count']
mean_count.plot()
plt.show()

# Q11.
final_data = data_time_categories.copy()

final_data["Hour Cat 1"] = (3 <= final_data['Hour']) & (final_data['Hour'] < 7)
final_data["Hour Cat 2"] = (7 <= final_data['Hour']) & (final_data['Hour'] < 10)
final_data["Hour Cat 3"] = (10 <= final_data['Hour']) & (final_data['Hour'] < 14)
final_data["Hour Cat 4"] = (14 <= final_data['Hour']) & (final_data['Hour'] < 22)
final_data["Hour Cat 5"] = (22 <= final_data['Hour']) | (final_data['Hour'] < 3)


# Q12.
def prediction_error(cols: list):

    X = final_data[cols].values.reshape(-1, len(cols))
    y = final_data['Rented Bike Count'].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    error = float(mean_squared_error(y, model.predict(X)))

    return error


prediction_error(["Temperature(°C)"])

# Q13.
model1 = prediction_error(['Hour'])
model2 = prediction_error(['Hour Cat 1', 'Hour Cat 2', 'Hour Cat 3', 'Hour Cat 4', 'Hour Cat 5'])

# Q14.
full_model_original = prediction_error([
    'Hour', 'Temperature(°C)', 'Wind speed (m/s)', 'Visibility (10m)', 'Dew point temperature(°C)',
    'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)'
])

full_model_updated = prediction_error([
    'Hour Cat 1', 'Hour Cat 2', 'Hour Cat 3', 'Hour Cat 4', 'Hour Cat 5', 'Temperature(°C)', 'Wind speed (m/s)',
    'Zero Solar Radiation', 'Zero Snowfall', 'Zero Rainfall', 'Max Visibility', 'Dew point temperature(°C)',
    'Spring', 'Summer', 'Autumn', 'Winter', 'Morning', 'Afternoon', 'Evening', 'Night', 'DayCount'
])

print(full_model_original, full_model_updated)
