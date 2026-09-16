import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score,recall_score,f1_score,roc_auc_score

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
