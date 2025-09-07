import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Download latest version
path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")

print("Path to dataset files:", path)
file_path = path +'/diabetes.csv'

dataset = pd.read_csv(file_path)
print(dataset.head())