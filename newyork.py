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
