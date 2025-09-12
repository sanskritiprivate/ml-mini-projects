import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Download latest version
path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")

print("Path to dataset files:", path)
file_path = path +'/diabetes.csv'

df = pd.read_csv(file_path)
print(df.head())
print(df.info())     # column names, data types, missing values
print(df.describe())  # summary statistics
print(df.isnull().sum())  # count missing values

# There are no null values, however, there are a lot of 0's that need
#  to be replaced by null values and taken care of.
cols_with_invalid_zeros = ["Glucose", "BloodPressure", "SkinThickness", "BMI", "Insulin"]
df[cols_with_invalid_zeros] = df[cols_with_invalid_zeros].replace(0, np.nan)

# then we replace all the nan values with the median
df.fillna(df.median(), inplace=True)

# verify after cleaning
print(df.isnull().sum())  # should be all zeros now
print(df.head())
