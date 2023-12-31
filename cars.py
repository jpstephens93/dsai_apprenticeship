import pandas as pd

df = pd.read_csv('data/cars.csv')
df.head()

# 1) mean weight
mean_weight = df['weight'].mean()

# 2) max hp
max_horsepower = df['horsepower'].max()

# 3) heavy cars
heavy_cars = df[df['weight'] > 3500]

# 4) df_ratio
df_ratio = df.copy()
df_ratio['ratio'] = df_ratio['horsepower'] / df_ratio['weight']

# 5) df_usa
df_usa = df.copy()
df_usa = df_usa[df_usa['origin'] == 'usa']

# 6) mean mpg usa
mean_mpg_usa = df_usa['mpg'].mean()

# 7) cylinders
eight_cyl_usa = len(df_usa[df_usa['cylinders'] == 8])

# 8) df hp
df_horsepower = df.copy()
df_horsepower = df_horsepower.dropna().reset_index(drop=True)

# 9) mode hp
mode_hp = df_horsepower['horsepower'].mode()[0]

# 10) df_high_hp
df_high_hp = df_horsepower.copy()
df_high_hp = df_high_hp[df_high_hp['horsepower'] >= mode_hp]

# 11) pct cyl
percentage_eight_cyl = 100 * (len(df_high_hp[df_high_hp['cylinders'] == 8]) / len(df_high_hp))

# 12) names
df_name = df.copy()
df_name['name_year'] = df_name['name'] + ' - 19' + df_name['model_year'].astype('str')

# 13) index
df_car_index = df_name.copy()
df_car_index.set_index('name_year', inplace=True)


# 14) acceleration
def acceleration(name_year):
    return df_car_index.loc[name_year, 'acceleration']


acceleration('ford torino - 1970')
