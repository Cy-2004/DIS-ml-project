import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('brain_tumor_dataset.csv')

# plt.figure(figsize=(10,6))
# sns.countplot(data=df, x='Tumor Type', hue='Gender', order=df['Tumor Type'].value_counts().index)
# plt.title('Most Common Tumor Types by Gender', fontsize=14)
# plt.xlabel('Tumor Type')
# plt.ylabel('Count')
# plt.xticks(rotation=45)
# plt.legend(title='Gender')
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10,6))
# sns.countplot(data=df, x='Tumor Type', order=df['Tumor Type'].value_counts().index, color='skyblue')
# plt.title('Distribution of Tumor Types (All Patients)', fontsize=14)
# plt.xlabel('Tumor Type')
# plt.ylabel('Count')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# Compute average tumor size per tumor type
avg_size = df.groupby('Tumor Type')['Size (cm)'].mean().sort_values(ascending=False)

# plt.figure(figsize=(10,6))
# sns.barplot(x=avg_size.index, y=avg_size.values, palette='coolwarm')
# plt.title('Average Tumor Size by Tumor Type', fontsize=14)
# plt.xlabel('Tumor Type')
# plt.ylabel('Average Size (cm)')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

print(df.head())
print(df.describe())
print(df.info())

print("Average size per tumor type:")
print(avg_size)

print("\nTumor counts:")
print(df['Tumor Type'].value_counts())

print("\nTumor counts by gender:")
print(df.groupby(['Tumor Type', 'Gender']).size().unstack(fill_value=0))

print("Missing values per column:\n", df.isnull().sum(), "\n")

# Feature correlation
# corr_matrix = df.corr()
# corr = corr_matrix["Tumor Type"].sort_values(ascending=False)
# print(corr)

# Split data into features and label
X = df.drop(columns=["Tumor Type"])
y = df["Tumor Type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Encode categorical variables
categorical_cols = ['Location', 'Grade', 'Gender']

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    encoders[col] = le

# Encode label
ley = LabelEncoder()
y_train_enc = ley.fit_transform(y_train)
y_test_enc = ley.transform(y_test)

# Scale numerical variables
numerical_cols = ['Size (cm)', 'Patient Age']

for col in numerical_cols:
    scaler = MinMaxScaler()
    X_train[col] = scaler.fit_transform(X_train[[col]])
    X_test[col] = scaler.transform(X_test[[col]])

# Train Random Forest
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train_enc)

# Evaluate
print("Accuracy:", random_forest.score(X_test, y_test_enc))

# Cross-validation
scores = cross_val_score(RandomForestClassifier(random_state=42), X_train, y_train_enc, cv=5)

print("CV mean:", scores.mean())
print("CV std:", scores.std())

# Feature correlation
corr_matrix = df.corr()
corr = corr_matrix["Tumor Type"].sort_values(ascending=False)
print(corr)

# Predictions
y_pred_enc = random_forest.predict(X_test)
y_pred = ley.inverse_transform(y_pred_enc)
y_actual = ley.inverse_transform(y_test_enc)

print("Decoded predictions:", y_pred[:20])
print("Decoded actual:", y_actual[:20])
