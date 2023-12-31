import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('data/AAPL.csv')
df.head()

df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.day_name()
df['Change %'] = (df['Adj Close'].pct_change() * 100)

# 1) adj close
mean_adj_close = df['Adj Close'].mean()

# 2) min low
min_low = df['Low'].min()

# 3) max high
max_high = df['High'].max()

# 4) price range
price_range = max_high - min_low

# 5) no rows
entries = df.shape[0]  # or len(df)

# 6) days of positive returns
mask = df['Change %'] > 0
positive_days = len(df[mask])

# 7) adj close greater than final value
final_value = df.loc[entries - 1, 'Adj Close']
mask = df['Adj Close'] > final_value
days_higher = len(df[mask])

# 8) df_2020
df_2020 = df[df['Year'] == 2020]
df_2020.set_index('Date', inplace=True)

# 9) mean_change_mon
mean_change_mon_2020 = df_2020[df_2020['Weekday'] == 'Monday']['Change %'].mean()

# 10) volume
total_volume_march_2020 = df_2020[df_2020['Month'] == 3]['Volume'].sum()

# 11) year high timestamp
year_high_timestamp = df_2020.idxmax()['Adj Close']

# 12) top 10 entries
df_top_10 = df.copy()
df_top_10 = df_top_10.sort_values('Change %', ascending=False).head(10)

# 13) not mondays
top_10_not_mon = len(df_top_10[df_top_10['Weekday'] != 'Monday'])

# 14) df variation
df_var = df.copy()
df_var['Variation %'] = ((df_var['High'] - df_var['Low']) / df_var['Close']) * 100

# 15) variation value
df_var_value = df_var.copy()
df_var_value['Traded Value'] = df_var_value['Volume'] * df_var_value['Adj Close']
