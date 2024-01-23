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

data_humidity['Humidity(%)'].str.replace('%', '')
