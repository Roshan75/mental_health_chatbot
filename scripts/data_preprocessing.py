import pandas as pd
import os

df = pd.read_csv('datasets/Combined_Data.csv')
print(df.head())
print(sum(df['statement'].isnull()))
df = df.dropna()
print(df.isnull().sum())
print(df['status'].unique(), df['status'].nunique())
# df_new = df['statement'].dropna()
# print(sum(df_new['statement'].isna()))