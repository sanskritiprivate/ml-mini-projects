# Classification Example: Titanic Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load Titanic dataset from Kaggle (CSV should be in same folder)
data = pd.read_csv("titanic.csv")

# Select features (simplified for starter code)
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Handle missing values
data = data[features + ["Survived"]].dropna()

# Convert categorical to numeric (Sex, Embarked)
data = pd.get_dummies(data, columns=["Sex", "Embarked"], drop_first=True)

X = data.drop("Survived", axis=1)
y = data["Survived"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Decision Tree ---
dtree = DecisionTreeClassifier(max_depth=4, random_state=42)
dtree.fit(X_train, y_train)
y_pred_dt = dtree.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# # Visualize tree
# plt.figure(figsize=(14, 8))
# plot_tree(dtree, feature_names=X.columns, class_names=["Died", "Survived"], filled=True)
# plt.show()
#
# # --- Random Forest ---
# rforest = RandomForestClassifier(n_estimators=100, random_state=42)
# rforest.fit(X_train, y_train)
# y_pred_rf = rforest.predict(X_test)
# print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
