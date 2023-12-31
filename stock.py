import pandas as pd

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
entries = df.shape[0]

# 6) days of positive returns

