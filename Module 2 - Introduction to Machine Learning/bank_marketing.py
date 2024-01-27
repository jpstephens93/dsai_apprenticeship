import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("Module 2 - Introduction to Machine Learning/data/bank-full.csv", sep=";")
df.head()

df.info()

# Q1.
df.isna().any()

# Q2.
df.isna().any().any()

# Q3.
sns.countplot(df, y='y')
plt.show()

# Q4.
sns.countplot(df, y='y', hue='marital')
plt.show()

# Q5.
df.boxplot('balance')
plt.show()
print("Can conclude that the majority of data points are clustered around / close to 0, with outliers "
      "skewing in a non-normally distributed way.")

# Q6.
sns.boxplot(df, x='y', y='balance')
plt.show()
print("Seems like non-married are more dispersed than married, so they can either do much better or much worse")

# Q7.
df_new = df.copy()
df_new['duration'] = df_new['duration'] / 60

# Q8.
df_numerical = df_new.copy()

job_df = pd.get_dummies(df_numerical['job'], dtype=int, prefix='job')
education_df = pd.get_dummies(df_numerical['education'], dtype=int, prefix='education')
df_numerical['y'] = pd.get_dummies(df_numerical['y'], dtype=int, drop_first=True)

df_numerical = pd.concat([df_numerical, job_df, education_df], axis=1)

# Q9.
df_numerical_education = df_numerical.copy()[
      ['education_primary', 'education_secondary', 'education_tertiary', 'education_unknown', 'balance', 'duration',
       'campaign', 'pdays', 'previous', 'y']
]

# Q10.
df_scaled = (df_numerical_education - df_numerical_education.min(axis=0)) / (df_numerical_education.max(axis=0) -
                                                                             df_numerical_education.min(axis=0))

# Q11.
X_train, X_test, y_train, y_test = train_test_split(
    df_scaled[list(df_scaled.columns[:-1])], df_scaled['y'], test_size=0.3, random_state=101
)

# Q12.
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

# Q13.
predictions = logmodel.predict(X_test)
accuracy_score(y_test, predictions)
