import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns


df = pd.read_csv("cleaned_facebook_liverpool.csv")


# EDA
print(df.info())

print(df.describe())

print(df.nunique())

print(df.isnull().sum())

print(df.dtypes)

print(df.corr())

print(df["no. of comments"].mode())

print(df.head())



# Viz
sns.heatmap(df.corr())
plt.show()

sns.lineplot(x=df["no. of likes"], y=df["date"])
plt.show()

sns.barplot(x=df["no. of comments"], y=df["date"])
plt.show()

sns.boxplot(x=df["no. of likes"])
plt.show()

sns.violinplot(x=df["no. of likes"])
plt.show()

sns.histplot(x=df["no. of likes"])
plt.show()