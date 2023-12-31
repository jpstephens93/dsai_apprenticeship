import pandas as pd

df = pd.read_csv('data/FTSE100.csv')
df.head()

# 1) tidy data
df_clean = df.copy()

mask = df_clean['Ticker'] == 'RDSA'
rdsa_index = df_clean[mask].index

df_clean.drop(rdsa_index, inplace=True)
df_clean.drop(columns='Strong Buy', inplace=True)

# 2) col data type
price_df = df_clean.copy()
price_df['Mid-price (p)'] = price_df['Mid-price (p)'].str.replace(',', '').astype(float)


# 3) format change values
# (a)
def format_change(string):
    if string[-1] == '%':
        return float(string[:-1])
    else:
        return float(string) * 100


string = '-0.015'
format_change(string)

# (b)
change_df = price_df.copy()
change_df['Change (%)'] = change_df['Change'].apply(format_change)

# 4) holding summary dict
holding = ('BP.', 500, 1535.21)

print(holding[0])
print(holding[1])
print(holding[2])

holding_dict = {
    'holding_cost': holding[1] * holding[2],
    'holding_value': holding[1] * change_df[change_df['Ticker'] == holding[0]]['Mid-price (p)'].values[0],
    'change_in_value': 100 * ((
        (holding[1] * change_df[change_df['Ticker'] == holding[0]]['Mid-price (p)'].values[0]) / (holding[1] * holding[2])
    ) - 1)
}

# 5) market comparison
comparison_df = change_df.copy()
avg_mkt_change = comparison_df['Change (%)'].mean()
comparison_df['Beat Market'] = comparison_df['Change (%)'] > avg_mkt_change
comparison_df['Buy Ratio'] = comparison_df['Buy'] / comparison_df['Brokers']

# 6) investigate
watchlist = [('TUI', 820.0), ('Whitbread', 4300.0), ('AstraZeneca', 7500.0),
             ('Standard Chartered', 920.0), ('Barclays', 135.5)]

companies_list = []
for company, price in watchlist:
    # print([company, price])
    company_row = comparison_df[comparison_df['Company'] == company]
    if price <= company_row['Mid-price (p)'].values[0] or company_row['Buy Ratio'].values[0] >= 0.5:
        companies_list.append(company)

