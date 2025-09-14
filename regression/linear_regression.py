import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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

# From Glucose, BMI, and Ubsulin, I picked Glucose to predict.
# These have quantitative values, and that's what we want since it's Linear regression
# as opposed to Logistic regression

# Drop outcome and glucose since glucose is the target and outcome is outcome, the rest
# are features
X = df.drop(columns=["Glucose", "Outcome"])
y= df["Glucose"]

# train-test split 80% train 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# what is random_state???

model = LinearRegression()
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

# evaluate model performance
print("R2 score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Visualization using matplotlib
# continue here later