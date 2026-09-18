# ============================================================
# NYC Airbnb - XGBoost Multiclass Classification + GridSearchCV
# ============================================================
# Target: room_type
#
# IMPORTANT:
# XGBoost multiclass classification requires integer target labels.
# LabelEncoder converts:
#   0 -> Entire home/apt
#   1 -> Private room
#   2 -> Shared room
#
# GridSearch:
#   1,458 parameter combinations
#   5-fold Stratified CV
#   7,290 CV fits
#   F1 Macro scoring
#   verbose=3
#   class balancing with sample_weight
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder
)

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 2. LOAD DATASET
# ============================================================
# For Google Colab:
# Upload AB_NYC_2019.csv to the Colab Files panel and use:
#
# df = pd.read_csv("AB_NYC_2019.csv")
#
# If you are running on your Windows PC, replace this with
# your Windows path.

df = pd.read_csv("AB_NYC_2019.csv")

print(df.head())
print(df.info())
print(df.describe())
print("Shape:", df.shape)


# ============================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Variable Distribution:")
print(df["room_type"].value_counts())

plt.figure(figsize=(8, 5))
sns.countplot(x="room_type", data=df)
plt.xticks(rotation=15)
plt.title("Room Type Distribution")
plt.show()


# ============================================================
# 4. UNIVARIATE ANALYSIS
# ============================================================

numerical_cols_eda = [
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365"
]

df[numerical_cols_eda].hist(
    bins=30,
    figsize=(12, 8)
)

plt.tight_layout()
plt.show()


# ============================================================
# 5. CATEGORICAL FEATURE ANALYSIS
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="neighbourhood_group"
)

plt.title("Neighbourhood Group Distribution")
plt.show()


# ============================================================
# 6. BIVARIATE ANALYSIS
# ============================================================

plt.figure(figsize=(9, 5))

sns.boxplot(
    x="room_type",
    y="price",
    data=df
)

plt.title("Price vs Room Type")
plt.xticks(rotation=15)
plt.show()


# ============================================================
# 7. CORRELATION
# ============================================================

corr = df[
    numerical_cols_eda + ["latitude", "longitude"]
].corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")
plt.show()


# ============================================================
# 8. GEOGRAPHIC DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 7))

sns.scatterplot(
    x="longitude",
    y="latitude",
    hue="room_type",
    data=df,
    alpha=0.5
)

plt.title("Geographic Distribution of Room Types")
plt.show()


# ============================================================
# 9. DATA CLEANING
# ============================================================

# Drop identifier/free-text columns
df_clean = df.drop(
    columns=[
        "id",
        "name",
        "host_id",
        "host_name",
        "last_review"
    ]
)

# No reviews yet -> 0 reviews per month
df_clean["reviews_per_month"] = (
    df_clean["reviews_per_month"].fillna(0)
)

# Cap extreme outliers at 99th percentile
price_cap = df_clean["price"].quantile(0.99)
nights_cap = df_clean["minimum_nights"].quantile(0.99)

df_clean["price"] = df_clean["price"].clip(
    upper=price_cap
)

df_clean["minimum_nights"] = df_clean[
    "minimum_nights"
].clip(
    upper=nights_cap
)

print("\nCleaned Dataset Shape:")
print(df_clean.shape)


# ============================================================
# 10. SEPARATE X AND y
# ============================================================

X = df_clean.drop(
    columns=["room_type"]
)

y = df_clean["room_type"]

print("\nOriginal Target Classes:")
print(y.unique())


# ============================================================
# 11. ENCODE TARGET VARIABLE
# ============================================================
# THIS FIXES THE PREVIOUS XGBOOST ERROR:
#
# ValueError:
# Invalid classes inferred from unique values of y
#
# XGBoost expects integer classes for this multiclass setup.

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nTarget Class Mapping:")

for number, name in enumerate(label_encoder.classes_):
    print(f"{number} -> {name}")

print("\nEncoded Target Classes:")
print(np.unique(y_encoded))


# ============================================================
# 12. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.33,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)

print("\nTraining Class Distribution:")
print(
    pd.Series(y_train)
    .value_counts()
    .sort_index()
)


# ============================================================
# 13. PREPROCESSING
# ============================================================

numerical_cols = [
    "latitude",
    "longitude",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365"
]

categorical_cols = [
    "neighbourhood_group",
    "neighbourhood"
]


# Numeric pipeline
numeric_pipeline = Pipeline(
    steps=[
        (
            "impute",
            SimpleImputer(strategy="median")
        ),
        (
            "scale",
            StandardScaler()
        )
    ]
)


# Categorical pipeline
categorical_pipeline = Pipeline(
    steps=[
        (
            "impute",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encode",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


# Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numeric_pipeline,
            numerical_cols
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_cols
        )
    ]
)

print("\nPreprocessor:")
print(preprocessor)


# ============================================================
# 14. HANDLE CLASS IMBALANCE
# ============================================================

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

print("\nSample weights calculated.")
print("Number of sample weights:", len(sample_weights))


# ============================================================
# 15. XGBOOST PIPELINE
# ============================================================

xgb_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            XGBClassifier(
                objective="multi:softmax",
                num_class=3,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=1
            )
        )
    ]
)

print("\nXGBoost Pipeline Created.")


# ============================================================
# 16. PARAMETER GRID
# ============================================================
# Total:
# 3 × 3 × 3 × 3 × 3 × 3 × 2
# = 1,458 combinations

param_grid = {

    "classifier__n_estimators": [
        200,
        300,
        400
    ],

    "classifier__max_depth": [
        3,
        5,
        7
    ],

    "classifier__learning_rate": [
        0.03,
        0.05,
        0.1
    ],

    "classifier__min_child_weight": [
        1,
        3,
        5
    ],

    "classifier__subsample": [
        0.8,
        0.9,
        1.0
    ],

    "classifier__colsample_bytree": [
        0.8,
        0.9,
        1.0
    ],

    "classifier__gamma": [
        0,
        0.1
    ]
}


# ============================================================
# 17. CALCULATE SEARCH SIZE
# ============================================================

total_combinations = (
    3 * 3 * 3 * 3 * 3 * 3 * 2
)

total_cv_fits = total_combinations * 5

print("\n" + "=" * 60)
print("GRID SEARCH SIZE")
print("=" * 60)

print(
    "Total Parameter Combinations:",
    total_combinations
)

print(
    "Total 5-Fold CV Fits:",
    total_cv_fits
)

print(
    "Final Refit:",
    1
)

print(
    "Total Training Operations:",
    total_cv_fits + 1
)


# ============================================================
# 18. 5-FOLD STRATIFIED CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\n5-Fold Stratified CV Created.")


# ============================================================
# 19. GRIDSEARCHCV
# ============================================================

xgb_grid = GridSearchCV(
    estimator=xgb_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=3,
    refit=True
)

print("\nGridSearchCV Created.")
print("Ready to start training.")


# ============================================================
# 20. START TRAINING
# ============================================================
# IMPORTANT:
# This is the long-running cell if you split the code into cells.
#
# 1,458 combinations × 5 folds = 7,290 CV fits.

xgb_grid.fit(
    X_train,
    y_train,
    classifier__sample_weight=sample_weights
)

print("\n" + "=" * 60)
print("GRID SEARCH FINISHED SUCCESSFULLY")
print("=" * 60)


# ============================================================
# 21. BEST PARAMETERS
# ============================================================

print("\n" + "=" * 60)
print("BEST PARAMETERS")
print("=" * 60)

print(xgb_grid.best_params_)


# ============================================================
# 22. BEST CV MACRO-F1
# ============================================================

print("\n" + "=" * 60)
print("BEST CV MACRO-F1")
print("=" * 60)

print(xgb_grid.best_score_)


# ============================================================
# 23. TEST PREDICTION
# ============================================================

xgb_pred = xgb_grid.predict(
    X_test
)

print("\nPredictions generated.")
print("Number of predictions:", len(xgb_pred))


# ============================================================
# 24. EVALUATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    xgb_pred
)

precision = precision_score(
    y_test,
    xgb_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    xgb_pred,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    y_test,
    xgb_pred,
    average="macro",
    zero_division=0
)

f1_weighted = f1_score(
    y_test,
    xgb_pred,
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 60)
print("TEST SET EVALUATION")
print("=" * 60)

print("Accuracy:", accuracy)
print("Precision Macro:", precision)
print("Recall Macro:", recall)
print("F1 Macro:", f1_macro)
print("F1 Weighted:", f1_weighted)


# ============================================================
# 25. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        xgb_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# 26. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    xgb_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ============================================================
# 27. CONFUSION MATRIX VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("XGBoost Confusion Matrix")

plt.tight_layout()
plt.show()


# ============================================================
# 28. SAVE GRIDSEARCH RESULTS
# ============================================================
# This saves the result of every tested parameter combination.
# Run after GridSearchCV has finished.

results_df = pd.DataFrame(
    xgb_grid.cv_results_
)

results_df = results_df.sort_values(
    by="rank_test_score"
)

print("\nTop 20 Parameter Combinations:")

display(
    results_df[
        [
            "rank_test_score",
            "mean_test_score",
            "std_test_score",
            "params"
        ]
    ].head(20)
)

results_df.to_csv(
    "xgboost_gridsearch_results.csv",
    index=False
)

print(
    "\nSaved: xgboost_gridsearch_results.csv"
)
