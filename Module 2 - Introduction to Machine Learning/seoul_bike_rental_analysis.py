import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

plt.rcParams["figure.dpi"] = 120

data = pd.read_csv("Module 2 - Introduction to Machine Learning/data/seoul_bike_data.csv")

# Q1.
categorical = [0, 2, 4, 11, 12, 13]
continuous = [1, 3, 5, 6, 7, 8, 9, 10]

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
data_season["Winter"] = data_season['Seasons'].str.contains('Winter')
data_season["Autumn"] = data_season['Seasons'].str.contains('Autumn')

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
data_binary["Max Visibility"] = [True if x < 0.1 else False for x in data_binary["Solar Radiation (MJ/m2)"]]
