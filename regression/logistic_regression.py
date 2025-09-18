# logistic_regression_pima.py
# Full end-to-end example with detailed comments.
# Run this after you've downloaded the dataset (see two loading options below).

import os
import glob
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# scikit-learn imports
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
)
from sklearn.metrics import ConfusionMatrixDisplay

# for VIF and statsmodels (optional)
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# for saving model
import joblib

# --------------------------
# 1) LOAD THE DATA
# --------------------------
# Option A: you already ran `path = kagglehub.dataset_download(...)` and have `path`
# If you have a variable `path` pointing to the dataset folder, set it here:
# path = "/Users/you/Library/Caches/kagglehub/datasets/uciml/pima-indians-diabetes-database/..."
#
# Option B: put diabetes.csv in the current working directory and we'll find it.

def load_pima_csv_from_path(path=None):
    """
    Tries to locate a CSV in `path` or the current directory.
    Returns a pandas DataFrame.
    """
    # If user supplied path, search for csv files there
    if path:
        csvs = glob.glob(os.path.join(path, "*.csv"))
        if not csvs:
            # if there's a zip (some downloads are zip files), extract
            zips = glob.glob(os.path.join(path, "*.zip"))
            if zips:
                with zipfile.ZipFile(zips[0], "r") as z:
                    z.extractall(path)
                csvs = glob.glob(os.path.join(path, "*.csv"))
        if csvs:
            file_path = csvs[0]
        else:
            raise FileNotFoundError(f"No CSV found in provided path: {path}")
    else:
        # No path provided: look in cwd
        csvs = glob.glob("*.csv")
        if len(csvs) == 0:
            raise FileNotFoundError(
                "No CSV found in current working directory. Place 'diabetes.csv' here or pass `path`."
            )
        file_path = csvs[0]

    print(f"Loading CSV: {file_path}")
    df = pd.read_csv(file_path)
    return df

# --------- Change this to the folder returned by kagglehub.dataset_download if you want:
path = None   # e.g., path = "/Users/you/Library/Caches/kagglehub/datasets/uciml/pima-indians-diabetes-database/..."
# If you have `path` from kagglehub, assign it above. Otherwise leave None and put the CSV in cwd.

df = load_pima_csv_from_path(path)

# Quick peek
print("\nData head (first 5 rows):")
print(df.head())
print("\nData info:")
print(df.info())
print("\nClass distribution (Outcome):")
print(df['Outcome'].value_counts(), "\n")  # 0 = no diabetes, 1 = diabetes

# --------------------------
# 2) CLEANING: treat zeros as missing for certain columns
# --------------------------
# Known issue for this dataset: several physiologic variables use 0 to encode missing.
cols_invalid_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Some versions of the dataset might have slightly different capitalization; map robustly
cols_present = [c for c in cols_invalid_zero if c in df.columns]
print("Columns we'll treat 0 as missing:", cols_present)

# Replace invalid zeros with np.nan
df[cols_present] = df[cols_present].replace(0, np.nan)

print("\nMissing value counts after replacement:")
print(df.isnull().sum())

# --------------------------
# 3) Decide on imputation strategy
#    We'll use median imputation (robust to skew / outliers).
#    Optionally you can use KNNImputer or iterative imputer for better imputations.
# --------------------------
imputer = SimpleImputer(strategy="median")

# Separate features and target
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Fit imputer on entire dataset (alternatively fit on train only if you want strict pipeline)
X_imputed_arr = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed_arr, columns=X.columns)

# Verify no missing after imputation
print("\nMissing after median imputation (should be 0s):")
print(pd.DataFrame(X_imputed).isnull().sum())

# --------------------------
# 4) Check multicollinearity with VIF (optional but useful)
# --------------------------
# VIF requires no constant and a numpy array
vif_data = pd.DataFrame()
vif_data["feature"] = X_imputed.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_imputed.values, i) for i in range(X_imputed.shape[1])
]
print("\nVariance Inflation Factors (VIF):")
print(vif_data)

# --------------------------
# 5) Train-test split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")

# --------------------------
# 6) Build a scikit-learn pipeline
#    Steps:
#      - (1) imputer - we've already imputed once for VIF; but we will also include imputer in pipeline to
#                      make the pipeline end-to-end robust (fit_transform performed inside pipeline)
#      - (2) scaler - StandardScaler to give features mean 0 std 1 (often helps optimizer)
#      - (3) logistic regression - solver='liblinear' is a good default for small datasets
# --------------------------
pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(solver="liblinear", penalty="l2", random_state=42, max_iter=1000),
)

# Fit pipeline on training data
pipeline.fit(X_train, y_train)

# --------------------------
# 7) Predictions and probabilities
# --------------------------
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]  # probability of class 1 (diabetes)

# --------------------------
# 8) Evaluation metrics
# --------------------------
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
fig, ax = plt.subplots(figsize=(5,5))
disp.plot(ax=ax, cmap=plt.cm.Blues)
ax.set_title("Confusion Matrix")
plt.show()

# ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()

# Precision-Recall curve (useful for imbalanced problems)
precision, recall, _ = precision_recall_curve(y_test, y_proba)
avg_precision = average_precision_score(y_test, y_proba)
plt.figure(figsize=(6,4))
plt.plot(recall, precision, label=f"PR curve (AP = {avg_precision:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------
# 9) Cross-validated AUC (5-fold stratified)
# --------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = cross_val_score(pipeline, X_imputed, y, cv=cv, scoring="roc_auc")
print(f"\nCross-validated ROC AUC (5-fold): mean={cv_aucs.mean():.4f}, std={cv_aucs.std():.4f}")
print("Per-fold AUCs:", cv_aucs)

# --------------------------
# 10) Inspect coefficients and compute odds ratios
#     We need to access the LogisticRegression step in the pipeline.
# --------------------------
# get logistic regression object (named_steps works with make_pipeline names)
logreg = pipeline.named_steps["logisticregression"]
scaler = pipeline.named_steps["standardscaler"]
imputer_in_pipeline = pipeline.named_steps["simpleimputer"]

# Because the coefficients correspond to scaled features, we can show coefficients and odds ratios.
coef = logreg.coef_.flatten()
features = X.columns.tolist()
coef_df = pd.DataFrame({"feature": features, "coefficient": coef})
coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])
coef_df = coef_df.sort_values(key=lambda df: np.abs(df["coefficient"]), ascending=False)
print("\nCoefficients (on scaled features) and Odds Ratios:")
print(coef_df)

# If you want coefficients on original (unscaled) features, you can unscale them:
# scikit-learn's StandardScaler stores scale_ and mean_ for unscaling:
scale = scaler.scale_
mean = scaler.mean_
# transform to coefficients for original units (beta_unscaled = beta_scaled / scale)
beta_unscaled = coef / scale
coef_orig_df = pd.DataFrame({"feature": features, "coef_unscaled": beta_unscaled})
coef_orig_df["odds_ratio_unscaled"] = np.exp(coef_orig_df["coef_unscaled"])
print("\nApproximate coefficients on original units (interpret with caution):")
print(coef_orig_df.sort_values(by="coef_unscaled", key=lambda s: np.abs(s), ascending=False))

# --------------------------
# 11) OPTIONAL: Fit statsmodels Logit to get p-values and summary
#     We'll fit on the imputed (median) TRAIN set without scaling so the model converges
# --------------------------
X_train_imputed = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
X_train_sm = sm.add_constant(X_train_imputed)  # add intercept
logit_sm = sm.Logit(y_train, X_train_sm)
try:
    res = logit_sm.fit(disp=False)  # disp=False suppresses fitting messages
    print("\n=== statsmodels Logit summary (train set) ===")
    print(res.summary2())
except Exception as e:
    print("\nstatsmodels Logit failed to converge or had an issue:", e)
    print("You can try a different starting point or check separability of predictors.")

# --------------------------
# 12) Save the trained pipeline for later use
# --------------------------
model_file = "pima_logistic_pipeline.joblib"
joblib.dump(pipeline, model_file)
print(f"\nSaved trained pipeline to {model_file}")

# --------------------------
# 13) Small helper: prediction function for new raw samples
# --------------------------
def predict_new_samples(df_new):
    """
    df_new: pandas DataFrame with same columns as X (raw values, may include 0s for missing)
    returns: DataFrame with predicted probability and predicted class
    """
    # Ensure same columns and order
    df_new = df_new[X.columns]
    probs = pipeline.predict_proba(df_new)[:, 1]
    preds = pipeline.predict(df_new)
    return pd.DataFrame({"predicted_proba": probs, "predicted_class": preds}, index=df_new.index)

# Example usage:
# new_row = X.iloc[0:2]  # just demo: use two rows from original data (already imputed)
# print(predict_new_samples(new_row))

# --------------------------
# End of script
# --------------------------

# todo: break code down into functions for readability
# todo: read through code, redo logistic regression
# todo: read linear and logistic regression math in ESL
# todo: write tests for linear and logistic regression