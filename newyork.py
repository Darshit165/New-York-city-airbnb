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