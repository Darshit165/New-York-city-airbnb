import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,roc_curve,classification_report
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("F:\\project\\New york airbnb\\AB_NYC_2019.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.shape)

#Exploratory Data Analysis (EDA)
#We explore the data in the standard order every ML engineer follows:
#Missing values
#Univariate analysis (one variable at a time — distributions, skewness)
#Bivariate analysis (relationship between features and the target)
#Correlation between numeric features
#Outlier inspection
print(df.isnull().sum())
#Target Variable — room_type
print(df['room_type'].value_counts())   

#Visualizing the target variable
sns.countplot(x='room_type', data=df)
plt.show()
#The classes are imbalanced -> Shared Room is a Small Minority.
#Univariate Analysis — Numeric Features
numerical_cols = ["price", "minimum_nights", "number_of_reviews",
                 "reviews_per_month", "calculated_host_listings_count",
                 "availability_365"]

df[numerical_cols].hist(bins=30, figsize=(12, 8))
plt.show()
#Univariate Analysis — Categorical Features
sns.countplot(data=df, x='neighbourhood_group')
plt.show()
#Bivariate Analysis — Features vs Target
sns.boxplot(x='room_type', y='price', data=df)
plt.show()

#Correlation between numeric features
corr = df[numerical_cols+['latitude', 'longitude']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

#Geographic Distribution (Bonus Visual)
sns.scatterplot(x='longitude', y='latitude', hue='room_type', data=df, alpha=0.5)
plt.show()

#Data Cleaning & Feature Engineering
#Drop columns that are pure identifiers or free text and carry no generalizable signal for a tabular model (id, name, host_id, host_name, last_review).
#Fill missing reviews_per_month with 0 (no reviews yet).
#Cap extreme outliers in price and minimum_nights using percentile clipping, so a handful of data-entry errors (e.g. $10,000/night, 1,250 minimum nights) don't distort the model.
#Separate features (X) from the target (y).

#1. Drop Colunmns that dont help us.
df_clean = df.drop(columns=['id', 'name', 'host_id', 'host_name', 'last_review'])
#2. No reviews yet -> 0 reviews per month, not missing
df_clean['reviews_per_month'] = df_clean['reviews_per_month'].fillna(0)
#3. Cap Extreme outliers instead of deleting rows.
price_cap  = df_clean['price'].quantile(0.99)
nights_cap = df_clean['minimum_nights'].quantile(0.99)

df_clean['price'] = df_clean['price'].clip(upper=price_cap)
df_clean['minimum_nights'] = df_clean['minimum_nights'].clip(upper=nights_cap)
# df_clean = df_clean[df_clean['price'] < price_cap]
# df_clean = df_clean[df_clean['minimum_nights'] < nights_cap]
print(df_clean.shape)

#4. Separate features (X) from the target (y)
y = df_clean['room_type']
X = df_clean.drop(columns=['room_type'])
#Train / Test Split
#We hold out 20% of the data as a test set that is never touched during model selection or tuning — it is only used once, at the very end, to report the final, honest performance. stratify=y keeps the same class proportions in both splits, which matters because the target is imbalanced.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

#Preprocessing: ColumnTransformer + Pipeline

#Production ML code should never manually transform train/test data with separate lines — it's error-prone and leaks information. Instead we build a single, reusable ColumnTransformer:

#Numeric features → median imputation + standard scaling
#Categorical features → most-frequent imputation + one-hot encoding
#This transformer will be the first step of every model pipeline below, so preprocessing is learned only on training data and applied consistently everywhere (no data leakage).
numerical_cols = ["latitude","longitude","price", "minimum_nights", "number_of_reviews",
                 "reviews_per_month", "calculated_host_listings_count",
                 "availability_365"]

categorical_cols = ['neighbourhood_group', 'neighbourhood']

#1. Pipeline -> for Numeric Columns
numeric_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy="median")),
    ('scale', StandardScaler())
])


#2. Pipeline -> for Categorical Columns
categorical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy="most_frequent")),
    ('encode', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ("numerical", numeric_pipeline, numerical_cols),
    ("categorical", categorical_pipeline, categorical_cols )
])

print(preprocessor)

#Logistic Regression pipeline
#pipeline
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ))
])
# 5-FOLD CROSS VALIDATION
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
cv_scores = cross_val_score(
    lr_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring='f1_macro'
)
lr_cv_f1 = cv_scores.mean()
#TRAIN MODEL
lr_pipeline.fit(X_train, y_train)
#PREDICTION
lr_pred = lr_pipeline.predict(X_test)
#EVALUATION METRICS
lr_accuracy = accuracy_score(y_test, lr_pred)

lr_precision = precision_score(
    y_test,
    lr_pred,
    average='macro',
    zero_division=0
)
lr_recall = recall_score(
    y_test,
    lr_pred,
    average='macro',
    zero_division=0
)
lr_f1_macro = f1_score(
    y_test,
    lr_pred,
    average='macro',
    zero_division=0
)
lr_f1_weighted = f1_score(
    y_test,
    lr_pred,
    average='weighted',
    zero_division=0
)
#print result
print("=" * 60)
print("LOGISTIC REGRESSION - EVALUATION")
print("=" * 60)

print("5-Fold CV F1 Macro:", lr_cv_f1)
print("Test Accuracy:", lr_accuracy)
print("Test Precision Macro:", lr_precision)
print("Test Recall Macro:", lr_recall)
print("Test F1 Macro:", lr_f1_macro)
print("Test F1 Weighted:", lr_f1_weighted)
#CLASSIFICATION REPORT
print("\nClassification Report:")
print(classification_report(
    y_test,
    lr_pred,
    zero_division=0
))
#CONFUSION MATRIX
cm = confusion_matrix(y_test, lr_pred)
print("\nConfusion Matrix:")
print(cm)
#CONFUSION MATRIX PLOT
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=lr_pipeline.classes_,
    yticklabels=lr_pipeline.classes_
)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Logistic Regression - Confusion Matrix")
plt.show()

#Decision Tree with default training
dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(
        class_weight='balanced',
        random_state=42
    ))
])
# 5-FOLD CROSS VALIDATION
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
cv_scores = cross_val_score(
    dt_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring='f1_macro'
)
dt_cv_f1 = cv_scores.mean()
# TRAIN MODEL
dt_pipeline.fit(X_train, y_train)
# PREDICTION
dt_pred = dt_pipeline.predict(X_test)
# EVALUATION METRICS
dt_accuracy = accuracy_score(y_test, dt_pred)

dt_precision = precision_score(
    y_test,
    dt_pred,
    average='macro',
    zero_division=0
)
dt_recall = recall_score(
    y_test,
    dt_pred,
    average='macro',
    zero_division=0
)
dt_f1_macro = f1_score(
    y_test,
    dt_pred,
    average='macro',
    zero_division=0
)
dt_f1_weighted = f1_score(
    y_test,
    dt_pred,
    average='weighted',
    zero_division=0
)
# PRINT RESULTS
print("=" * 60)
print("DECISION TREE - EVALUATION")
print("=" * 60)

print("5-Fold CV F1 Macro:", dt_cv_f1)
print("Test Accuracy:", dt_accuracy)
print("Test Precision Macro:", dt_precision)
print("Test Recall Macro:", dt_recall)
print("Test F1 Macro:", dt_f1_macro)
print("Test F1 Weighted:", dt_f1_weighted)
# CLASSIFICATION REPORT
dt_cm = confusion_matrix(y_test, dt_pred)
print("\nClassification Report:")
print(classification_report(
    y_test,
    dt_pred,
    zero_division=0
))
# CONFUSION MATRIX
plt.figure(figsize=(8, 6))
sns.heatmap(
    dt_cm,
    annot=True,
    fmt='d',
    xticklabels=dt_pipeline.classes_,
    yticklabels=dt_pipeline.classes_
)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Decision Tree - Confusion Matrix")
plt.show()

# Hyper parameter tuning for decision tree using gridsearch CV
from sklearn.model_selection import GridSearchCV
#PARAMETER GRID
param_grid = {
    'classifier__criterion': ['gini'],
    'classifier__max_depth': [22],
    'classifier__min_samples_split': [3],
    'classifier__min_samples_leaf': [1]
}
#5-FOLD STRATIFIED CROSS VALIDATION
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
#GRID SEARCH
dt_grid = GridSearchCV(
    estimator=dt_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=3
)
#TRAIN GRID SEARCH
dt_grid.fit(X_train, y_train)
#BEST PARAMETERS
print("Best Parameters:")
print(dt_grid.best_params_)

print("\nBest CV F1 Macro:")
print(dt_grid.best_score_)

# FINAL PREDICTION
dt_pred_grid = dt_grid.predict(X_test)

# EVALUATION METRICS
dt_accuracy = accuracy_score(y_test, dt_pred_grid)
dt_precision = precision_score(
    y_test,
    dt_pred_grid,
    average='macro',
    zero_division=0
)
dt_recall = recall_score(
    y_test,
    dt_pred_grid,
    average='macro',
    zero_division=0
)
dt_f1_macro = f1_score(
    y_test,
    dt_pred_grid,
    average='macro',
    zero_division=0
)
dt_f1_weighted = f1_score(
    y_test,
    dt_pred_grid,
    average='weighted',
    zero_division=0
)
#FINAL RESULTS
print("=" * 60)
print("TUNED DECISION TREE - EVALUATION")
print("=" * 60)
print("5-Fold CV F1 Macro:", dt_grid.best_score_)
print("Test Accuracy:", dt_accuracy)
print("Test Precision Macro:", dt_precision)
print("Test Recall Macro:", dt_recall)
print("Test F1 Macro:", dt_f1_macro)
print("Test F1 Weighted:", dt_f1_weighted)
# CLASSIFICATION REPORT
print("\nClassification Report:")
print(classification_report(
    y_test,
    dt_pred_grid,
    zero_division=0
))
# CONFUSION MATRIX
dt_cm = confusion_matrix(y_test, dt_pred_grid)
print("\nConfusion Matrix:")
print(dt_cm)
# CONFUSION MATRIX PLOT
plt.figure(figsize=(8, 6))
sns.heatmap(
    dt_cm,
    annot=True,
    fmt='d',
    xticklabels=dt_grid.classes_,
    yticklabels=dt_grid.classes_
)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Tuned Decision Tree - Confusion Matrix")
plt.show()


#Random forest defualt parameter
# PIPELINE
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])
# 5-FOLD CROSS VALIDATION
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
cv_scores = cross_val_score(
    rf_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1
)
rf_cv_f1 = cv_scores.mean()
# TRAIN MODEL
rf_pipeline.fit(X_train, y_train)
# PREDICTION
rf_pred = rf_pipeline.predict(X_test)
# EVALUATION METRICS
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(
    y_test,
    rf_pred,
    average='macro',
    zero_division=0
)
rf_recall = recall_score(
    y_test,
    rf_pred,
    average='macro',
    zero_division=0
)
rf_f1_macro = f1_score(
    y_test,
    rf_pred,
    average='macro',
    zero_division=0
)
rf_f1_weighted = f1_score(
    y_test,
    rf_pred,
    average='weighted',
    zero_division=0
)
# PRINT RESULTS
print("=" * 60)
print("RANDOM FOREST - EVALUATION")
print("=" * 60)
print("5-Fold CV F1 Macro:", rf_cv_f1)
print("Test Accuracy:", rf_accuracy)
print("Test Precision Macro:", rf_precision)
print("Test Recall Macro:", rf_recall)
print("Test F1 Macro:", rf_f1_macro)
print("Test F1 Weighted:", rf_f1_weighted)
# CONFUSION MATRIX
rf_cm = confusion_matrix(y_test, rf_pred)
print("\nConfusion Matrix:")
print(rf_cm)
# CONFUSION MATRIX PLOT
plt.figure(figsize=(8, 6))
sns.heatmap(
    rf_cm,
    annot=True,
    fmt='d',
    xticklabels=rf_pipeline.classes_,
    yticklabels=rf_pipeline.classes_
)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Random Forest - Confusion Matrix")
plt.show()

# Hyperparameter Tuning for Random Forest using GridSearchCV

param_grid = {
    'classifier__n_estimators': [300],
    'classifier__max_depth': [None],
    'classifier__min_samples_split': [5],
    'classifier__min_samples_leaf': [1],
    'classifier__max_features': ['sqrt'],
}
# 5-FOLD STRATIFIED CV
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
# GRID SEARCH
rf_grid = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=3,
    refit=True
)
# TRAIN
rf_grid.fit(X_train, y_train)
# BEST PARAMETERS
print("Best Parameters:")
print(rf_grid.best_params_)

print("\nBest CV Macro-F1:")
print(rf_grid.best_score_)

# FINAL PREDICTION
rf_pred_grid = rf_grid.predict(X_test)

# EVALUATION METRICS
rf_grid_accuracy = accuracy_score(y_test, rf_pred_grid)
rf_grid_precision = precision_score(
    y_test,
    rf_pred_grid,
    average='macro',
    zero_division=0
)
rf_grid_recall = recall_score(
    y_test,
    rf_pred_grid,
    average='macro',
    zero_division=0
)
rf_grid_f1_macro = f1_score(
    y_test,
    rf_pred_grid,
    average='macro',
    zero_division=0
)
rf_grid_f1_weighted = f1_score(
    y_test,
    rf_pred_grid,
    average='weighted',
    zero_division=0
)
#FINAL RESULTS
print("=" * 60)
print("TUNED RANDOM FOREST - EVALUATION")
print("=" * 60)
print("5-Fold CV F1 Macro:", rf_grid.best_score_)
print("Test Accuracy:", rf_grid_accuracy)
print("Test Precision Macro:", rf_grid_precision)
print("Test Recall Macro:", rf_grid_recall)
print("Test F1 Macro:", rf_grid_f1_macro)
print("Test F1 Weighted:", rf_grid_f1_weighted)
# CLASSIFICATION REPORT
print("\nClassification Report:")
print(classification_report(
    y_test,
    rf_pred_grid,
    zero_division=0
))
# CONFUSION MATRIX
rf_cm = confusion_matrix(y_test, rf_pred_grid)
print("\nConfusion Matrix:")
print(rf_cm)
# CONFUSION MATRIX PLOT
plt.figure(figsize=(8, 6))
sns.heatmap(
    rf_cm,
    annot=True,
    fmt='d',
    xticklabels=rf_grid.classes_,
    yticklabels=rf_grid.classes_
)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Tuned Random Forest - Confusion Matrix")
plt.show()

#train xg boost model with default parameters
# ============================================================
# XGBOOST - DEFAULT MODEL
# ============================================================

from sklearn.utils.class_weight import compute_sample_weight

# Class mapping
class_mapping = {
    'Entire home/apt': 0,
    'Private room': 1,
    'Shared room': 2
}

y_train_xgb = y_train.map(class_mapping)
y_test_xgb = y_test.map(class_mapping)

# Balanced sample weights
xgb_sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=y_train_xgb
)

print("Training class distribution:")
print(y_train_xgb.value_counts())

print("\nTest class distribution:")
print(y_test_xgb.value_counts())


# ------------------------------------------------------------
# DEFAULT XGBOOST PIPELINE
# ------------------------------------------------------------

xgb_default_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    ))
])


# ------------------------------------------------------------
# 5-FOLD STRATIFIED CROSS VALIDATION
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

xgb_default_cv_scores = cross_val_score(
    xgb_default_pipeline,
    X_train,
    y_train_xgb,
    cv=cv,
    scoring='f1_macro',
    params={
        'classifier__sample_weight': xgb_sample_weights
    }
)

xgb_default_cv_f1 = xgb_default_cv_scores.mean()

print("\n5-Fold CV F1 Macro Scores:")
print(xgb_default_cv_scores)

print("\nMean CV F1 Macro:")
print(xgb_default_cv_f1)


# ------------------------------------------------------------
# TRAIN DEFAULT XGBOOST
# ------------------------------------------------------------

xgb_default_pipeline.fit(
    X_train,
    y_train_xgb,
    classifier__sample_weight=xgb_sample_weights
)


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

xgb_default_pred = xgb_default_pipeline.predict(X_test)


# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------

xgb_default_accuracy = accuracy_score(
    y_test_xgb,
    xgb_default_pred
)

xgb_default_precision = precision_score(
    y_test_xgb,
    xgb_default_pred,
    average='macro',
    zero_division=0
)

xgb_default_recall = recall_score(
    y_test_xgb,
    xgb_default_pred,
    average='macro',
    zero_division=0
)

xgb_default_f1_macro = f1_score(
    y_test_xgb,
    xgb_default_pred,
    average='macro',
    zero_division=0
)

xgb_default_f1_weighted = f1_score(
    y_test_xgb,
    xgb_default_pred,
    average='weighted',
    zero_division=0
)


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DEFAULT XGBOOST - FINAL EVALUATION")
print("=" * 70)

print("5-Fold CV F1 Macro:", xgb_default_cv_f1)
print("Test Accuracy:", xgb_default_accuracy)
print("Test Precision Macro:", xgb_default_precision)
print("Test Recall Macro:", xgb_default_recall)
print("Test F1 Macro:", xgb_default_f1_macro)
print("Test F1 Weighted:", xgb_default_f1_weighted)

print("\nClassification Report:")
print(
    classification_report(
        y_test_xgb,
        xgb_default_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test_xgb,
        xgb_default_pred
    )
)


#tuning XGBoost hyperparameters can be done using GridSearchCV 

X_xgb_train, X_xgb_val, y_xgb_train, y_xgb_val = train_test_split(
    X_train,
    y_train_xgb,
    test_size=0.20,
    random_state=42,
    stratify=y_train_xgb
)

xgb_sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=y_xgb_train
)
xgb_tuned_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    ))
])

param_grid = {
    "classifier__n_estimators": [400],
    "classifier__max_depth": [7],
    "classifier__learning_rate": [0.1],
    "classifier__min_child_weight": [1],
    "classifier__subsample": [0.8],
    "classifier__colsample_bytree": [0.8],
    "classifier__gamma": [0.1]
}

from sklearn.model_selection import GridSearchCV, PredefinedSplit
import numpy as np

# Combine training and validation data
X_grid = pd.concat([X_xgb_train, X_xgb_val])
y_grid = pd.concat([y_xgb_train, y_xgb_val])

# -1 = training data
#  0 = validation data
test_fold = np.concatenate([
    np.full(len(X_xgb_train), -1),
    np.zeros(len(X_xgb_val))
])

ps = PredefinedSplit(test_fold=test_fold)

# Combine sample weights
grid_sample_weights = np.concatenate([
    xgb_sample_weights,
    np.zeros(len(X_xgb_val))
])

xgb_grid = GridSearchCV(
    estimator=xgb_tuned_pipeline,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=ps,
    n_jobs=1,
    verbose=3,
    refit=True
)

xgb_grid.fit(
    X_grid,
    y_grid,
    classifier__sample_weight=grid_sample_weights
)

print("Best parameters found: ", xgb_grid.best_params_)
print("Best cross-validation score: ", xgb_grid.best_score_)

xgb_pred_grid = xgb_grid.predict(X_test)

# EVALUATION METRICS
xgb_accuracy = accuracy_score(
    y_test_xgb,
    xgb_pred_grid
)

xgb_precision = precision_score(
    y_test_xgb,
    xgb_pred_grid,
    average='macro',
    zero_division=0
)

xgb_recall = recall_score(
    y_test_xgb,
    xgb_pred_grid,
    average='macro',
    zero_division=0
)

xgb_f1_macro = f1_score(
    y_test_xgb,
    xgb_pred_grid,
    average='macro',
    zero_division=0
)

xgb_f1_weighted = f1_score(
    y_test_xgb,
    xgb_pred_grid,
    average='weighted',
    zero_division=0
)


# FINAL RESULTS
print("=" * 60)
print("TUNED XGBOOST - EVALUATION")
print("=" * 60)

print("Validation F1 Macro:", xgb_grid.best_score_)
print("Test Accuracy:", xgb_accuracy)
print("Test Precision Macro:", xgb_precision)
print("Test Recall Macro:", xgb_recall)
print("Test F1 Macro:", xgb_f1_macro)
print("Test F1 Weighted:", xgb_f1_weighted)


# CLASSIFICATION REPORT
print("\nClassification Report:")
print(classification_report(
    y_test_xgb,
    xgb_pred_grid,
    zero_division=0
))


# CONFUSION MATRIX
xgb_cm = confusion_matrix(
    y_test_xgb,
    xgb_pred_grid
)

print("\nConfusion Matrix:")
print(xgb_cm)


# CONFUSION MATRIX PLOT
plt.figure(figsize=(8, 6))

sns.heatmap(
    xgb_cm,
    annot=True,
    fmt='d',
    xticklabels=class_mapping.keys(),
    yticklabels=class_mapping.keys()
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Tuned XGBoost - Confusion Matrix")

plt.show()


# ============================================================
# FINAL MODEL COMPARISON
# ============================================================

# Predictions

lr_final_pred = lr_pipeline.predict(X_test)

dt_default_pred = dt_pipeline.predict(X_test)

dt_tuned_pred = dt_grid.predict(X_test)

rf_default_pred = rf_pipeline.predict(X_test)

rf_tuned_pred = rf_grid.predict(X_test)

# XGBoost DEFAULT
xgb_default_pred = xgb_default_pipeline.predict(X_test)

# XGBoost TUNED
xgb_tuned_pred = xgb_grid.predict(X_test)


# ------------------------------------------------------------
# METRIC FUNCTION
# ------------------------------------------------------------

def calculate_metrics(y_true, y_pred):

    return {
        "Test Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "Test Precision Macro": precision_score(
            y_true,
            y_pred,
            average='macro',
            zero_division=0
        ),

        "Test Recall Macro": recall_score(
            y_true,
            y_pred,
            average='macro',
            zero_division=0
        ),

        "Test F1 Macro": f1_score(
            y_true,
            y_pred,
            average='macro',
            zero_division=0
        ),

        "Test F1 Weighted": f1_score(
            y_true,
            y_pred,
            average='weighted',
            zero_division=0
        )
    }


# ------------------------------------------------------------
# COMPARISON TABLE
# ------------------------------------------------------------

comparison_results = [

    {
        "Model": "Logistic Regression",
        "Balance Strategy": "class_weight='balanced'",
        "CV F1 Macro": lr_cv_f1,
        **calculate_metrics(y_test, lr_final_pred)
    },

    {
        "Model": "Decision Tree - Default",
        "Balance Strategy": "class_weight='balanced'",
        "CV F1 Macro": dt_cv_f1,
        **calculate_metrics(y_test, dt_default_pred)
    },

    {
        "Model": "Decision Tree - Tuned",
        "Balance Strategy": "class_weight='balanced'",
        "CV F1 Macro": dt_grid.best_score_,
        **calculate_metrics(y_test, dt_tuned_pred)
    },

    {
        "Model": "Random Forest - Default",
        "Balance Strategy": "class_weight='balanced'",
        "CV F1 Macro": rf_cv_f1,
        **calculate_metrics(y_test, rf_default_pred)
    },

    {
        "Model": "Random Forest - Tuned",
        "Balance Strategy": "class_weight='balanced'",
        "CV F1 Macro": rf_grid.best_score_,
        **calculate_metrics(y_test, rf_tuned_pred)
    },

    {
        "Model": "XGBoost - Default",
        "Balance Strategy": "Balanced sample_weight",
        "CV F1 Macro": xgb_default_cv_f1,
        **calculate_metrics(y_test_xgb, xgb_default_pred)
    },

    {
        "Model": "XGBoost - Tuned",
        "Balance Strategy": "Balanced sample_weight",
        "CV F1 Macro": xgb_grid.best_score_,
        **calculate_metrics(y_test_xgb, xgb_tuned_pred)
    }
]


comparison_df = pd.DataFrame(comparison_results)


# ------------------------------------------------------------
# SORT BY MACRO F1
# ------------------------------------------------------------

comparison_df = comparison_df.sort_values(
    by="Test F1 Macro",
    ascending=False
).reset_index(drop=True)


# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------

print("=" * 120)
print("FINAL MODEL COMPARISON")
print("=" * 120)

print(comparison_df.round(4).to_string(index=False))