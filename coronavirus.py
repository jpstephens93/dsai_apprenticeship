import pandas as pd

data = pd.read_csv('data/owid-covid-data.csv')
data['date'] = pd.to_datetime(data['date'])
df = data[data['date'] == '2020-07-01']

df.head()

# 1)
countries = df.copy()
countries = countries.drop(countries[countries['location'] == 'World'].index)

# 3)
cols = ['continent', 'location', 'total_deaths_per_million']

countries_dr = countries[cols].sort_values('total_deaths_per_million', ascending=False)

# 4)
africa_tests = int(countries[countries['continent'] == 'Africa']['total_tests'].sum())

# 5)
africa_missing_test_data = len(countries[(countries['continent'] == 'Africa') & (countries['total_tests'].isna())])

# 6)
uk_no_tests = countries[countries['location'] == 'United Kingdom']['total_tests'].values[0]
countries_more_tests = len(countries[countries['total_tests'] > uk_no_tests])

# 7)
beds_dr = countries[['hospital_beds_per_thousand', 'total_deaths_per_million']].dropna()

# 8)
mask = beds_dr['hospital_beds_per_thousand'] > beds_dr['hospital_beds_per_thousand'].mean()
dr_high_bed_ratio = beds_dr[mask]['total_deaths_per_million'].mean()

# 9)
mask = beds_dr['hospital_beds_per_thousand'] < beds_dr['hospital_beds_per_thousand'].mean()
dr_low_bed_ratio = beds_dr[mask]['total_deaths_per_million'].mean()

# 10)
no_new_cases = countries[countries['new_cases'] == 0]

# 11)
highest_no_new = no_new_cases[no_new_cases['total_cases'] == no_new_cases['total_cases'].max()]['location'].values[0]

# 12)
int(round(countries[countries['total_deaths'] == 0]['population'].sum() / 1e6))


# 13)
def country_metric(df, location, metric):
    return df[df['location'] == location].iloc[0][metric]


# 14)
vietnam_older_70 = country_metric(countries, 'Vietnam', 'aged_70_older')


# 15)
def countries_average(df: pd.DataFrame, countries_list: list, metric: str):
    return df[df['location'].isin(countries_list)][metric].mean()


# test
df = countries.copy()
countries_list = ['Vietnam', 'United Kingdom']
metric = 'life_expectancy'

# 16)
g7 = ['United States', 'Italy', 'Canada', 'Japan', 'United Kingdom', 'Germany', 'France']
g7_avg_life_expectancy = countries_average(df, g7, 'life_expectancy')

# 17)
min_life_exp = countries['life_expectancy'].min()
country = countries[countries['life_expectancy'] == min_life_exp]['location'].values[0]
diff = round(g7_avg_life_expectancy - min_life_exp, 1)

headline = f'{country} has a life expectancy of {diff} years lower than the G7 average.'
