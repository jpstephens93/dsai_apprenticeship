import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = pd.read_csv('Module 2 - Introduction to Machine Learning/data/rental_data.csv')

data.info()
data.describe()

# Drop rows with NaN in `review_scores_rating`
data = data[~data['review_scores_rating'].isna()]

data.describe(percentiles=[0.99])

g = sns.boxplot(data=data)

plt.xticks(rotation=90)
plt.show()


def zscore(sample):
    mean = sample.mean()
    std = sample.std()

    return (sample - mean) / std


to_drop = []

mask = (abs(zscore(data['bedrooms'])) > 3) | (abs(zscore(data['price'])) > 3) | (abs(zscore(data['square_feet'])) > 3)

for i in data[mask].index:
    to_drop.append(i)

data.drop(index=to_drop, inplace=True)

g = sns.boxplot(data=data)

plt.xticks(rotation=90)
plt.show()

# scaling
cont_vars = ['accommodates', 'bathrooms', 'bedrooms', 'price',
             'review_scores_rating', 'square_feet', 'number_of_reviews',
             'reviews_per_month']

stander = StandardScaler()

data_s = pd.DataFrame(stander.fit_transform(data[cont_vars]), columns=cont_vars)

g = sns.boxplot(data=data_s)
plt.show()

normer = MinMaxScaler()

data_n = pd.DataFrame(normer.fit_transform(data[cont_vars]), columns=cont_vars)

g = sns.boxplot(data=data_n)
plt.show()

# discrete
pd.get_dummies(data, columns=['host_neighbourhood'], prefix='hood')
print(data.host_neighbourhood.astype('category').cat.codes)
