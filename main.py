import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('brain_tumor_dataset_new.csv')

# plt.figure(figsize=(8,5))
# sns.countplot(data=df, x="Histology", order=df["Histology"].value_counts().index)
# plt.title("Histogram of Histology Counts")
# plt.xlabel("Histology Type")
# plt.ylabel("Count")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10,6))
# sns.countplot(data=df, x="Location", hue="Histology")
# plt.title("Histology Distribution by Tumor Location")
# plt.xlabel("Tumor Location")
# plt.ylabel("Count")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8,5))
# sns.boxplot(data=df, x="Histology", y="Tumor_Growth_Rate")
# plt.title("Tumor Growth Rate by Histology")
# plt.xlabel("Histology")
# plt.ylabel("Growth Rate")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8,5))
# sns.boxplot(data=df, x="Histology", y="Survival_Rate")
# plt.title("Survival Rate by Histology")
# plt.xlabel("Histology")
# plt.ylabel("Survival Rate")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8,5))
# sns.boxplot(data=df, x="Histology", y="Tumor_Size")
# plt.title("Tumor Size by Histology")
# plt.xlabel("Histology")
# plt.ylabel("Tumor Size")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# print("Histology counts:")
# print(df["Histology"].value_counts())

# print("\nHistology by location:")
# print(df.groupby("Location")["Histology"].value_counts())

# print("\nTumor growth rate:")
# print(df.groupby("Histology")["Tumor_Growth_Rate"].describe())

# print("\nSurvival rate:")
# print(df.groupby("Histology")["Survival_Rate"].describe())

# print("\nTumor Size:")
# print(df.groupby("Histology")["Tumor_Size"].describe())

# print(df.head())
# print(df.describe())
# print(df.info())

# print("Missing values per column:\n", df.isnull().sum(), "\n")





# # Drop unnecessary columns
# df = df.drop(columns=["Patient_ID", "Survival_Rate", "Follow_Up_Required"]) # Survival_Rate and Follow_Up_Required are possible data leaks

# # Split data into features and label
# X = df.drop(columns=["Histology"])
# y = df["Histology"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# categorical_cols = X_train.select_dtypes(include=['object']).columns
# numeric_cols = X_train.select_dtypes(include=['int64','float64']).columns

# # Encode categorical variables
# encoders = {}
# for col in categorical_cols:
#     le = LabelEncoder()
#     X_train[col] = le.fit_transform(X_train[col])
#     X_test[col] = le.transform(X_test[col])
#     encoders[col] = le

# # Encode label
# ley = LabelEncoder()
# y_train_enc = ley.fit_transform(y_train)
# y_test_enc = ley.transform(y_test)

# # Scale numerical variables
# for col in numeric_cols:
#     scaler = MinMaxScaler()
#     X_train[col] = scaler.fit_transform(X_train[[col]])
#     X_test[col] = scaler.transform(X_test[[col]])

# # Train Random Forest
# random_forest = RandomForestClassifier(random_state=42)
# random_forest.fit(X_train, y_train_enc)

# # Evaluate
# train_accuracy = random_forest.score(X_train, y_train_enc)
# print("Training Accuracy:", train_accuracy)

# test_accuracy = random_forest.score(X_test, y_test_enc)
# print("Test Accuracy:", test_accuracy)

# # Cross-validation
# scores = cross_val_score(random_forest, X_train, y_train_enc, cv=5)

# print("CV mean:", scores.mean())
# print("CV std:", scores.std())

# # Predictions
# y_pred_enc = random_forest.predict(X_test)
# y_pred = ley.inverse_transform(y_pred_enc)
# y_actual = ley.inverse_transform(y_test_enc)

# print("Decoded predictions:", y_pred[:20])
# print("Decoded actual:", y_actual[:20])

# # Get feature importances
# importances = random_forest.feature_importances_
# feature_importance_df = pd.DataFrame({'Feature': X_train.columns,'Importance': importances}).sort_values(by='Importance', ascending=False)
# print("Feature importance:", feature_importance_df)





# Drop unnecessary columns, Survival_Rate and Follow_Up_Required are possible data leaks
df = df.drop(columns=["Patient_ID", "Survival_Rate", "Follow_Up_Required"])

# # Split data into features and label
# X = df.drop(columns=["Histology"])
# y = df["Histology"]

# categorical_cols = X.select_dtypes(include=['object']).columns
# numeric_cols = X.select_dtypes(include=['int64','float64']).columns

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Scale numerical columns
# scaler = MinMaxScaler()
# X_train_num = pd.DataFrame(scaler.fit_transform(X_train[numeric_cols]),columns=numeric_cols,index=X_train.index)
# X_test_num = pd.DataFrame(scaler.transform(X_test[numeric_cols]),columns=numeric_cols,index=X_test.index)

# # One-Hot Encode categorical
# ohe = OneHotEncoder(sparse_output=False, drop='first')
# X_train_cat = pd.DataFrame(ohe.fit_transform(X_train[categorical_cols]),columns=ohe.get_feature_names_out(categorical_cols),index=X_train.index)
# X_test_cat = pd.DataFrame(ohe.transform(X_test[categorical_cols]),columns=ohe.get_feature_names_out(categorical_cols),index=X_test.index)

# # Combine numeric + categorical
# X_train_enc = pd.concat([X_train_num, X_train_cat], axis=1)
# X_test_enc = pd.concat([X_test_num, X_test_cat], axis=1)

# # Encode label
# ley = LabelEncoder()
# y_train_enc = ley.fit_transform(y_train)
# y_test_enc = ley.transform(y_test)

# # Train Random Forest
# random_forest = RandomForestClassifier(
#     n_estimators=200,   # more trees
#     max_depth=5,        # limit depth
#     min_samples_leaf=5, # each leaf has at least 5 samples
#     max_features='sqrt', # consider subset of features for splits
#     random_state=42
# )
# random_forest.fit(X_train_enc, y_train_enc)

# # Evaluate
# train_accuracy = random_forest.score(X_train_enc, y_train_enc)
# print("Training Accuracy:", train_accuracy)

# test_accuracy = random_forest.score(X_test_enc, y_test_enc)
# print("Test Accuracy:", test_accuracy)

# # Cross-validation
# scores = cross_val_score(random_forest, X_train_enc, y_train_enc, cv=5)

# print("CV mean:", scores.mean())
# print("CV std:", scores.std())

# # Predictions
# y_pred_enc = random_forest.predict(X_test_enc)
# y_pred = ley.inverse_transform(y_pred_enc)
# y_actual = ley.inverse_transform(y_test_enc)

# # print("Decoded predictions:", y_pred[:20])
# # print("Decoded actual:", y_actual[:20])

# # Get feature importances
# importances = random_forest.feature_importances_
# feature_importance_df = pd.DataFrame({'Feature': X_train_enc.columns,'Importance': importances}).sort_values(by='Importance', ascending=False)
# print("Feature importance:", feature_importance_df)

# # Identify columns to keep
# cols_to_keep = feature_importance_df.head(5)["Feature"].tolist()

# X_train_enc = X_train_enc[cols_to_keep]
# X_test_enc = X_test_enc[cols_to_keep]

# print("Columns kept for model:", X_train_enc.columns.tolist())

# # Retrain Random Forest
# random_forest = RandomForestClassifier(
#     n_estimators=200,   # more trees
#     max_depth=5,        # limit depth
#     min_samples_leaf=5, # each leaf has at least 5 samples
#     max_features='sqrt', # consider subset of features for splits
#     random_state=42
# )
# random_forest.fit(X_train_enc, y_train_enc)

# # Evaluate
# train_accuracy = random_forest.score(X_train_enc, y_train_enc)
# print("Training Accuracy:", train_accuracy)

# test_accuracy = random_forest.score(X_test_enc, y_test_enc)
# print("Test Accuracy:", test_accuracy)

# # Cross-validation
# scores = cross_val_score(random_forest, X_train_enc, y_train_enc, cv=5)
# print("CV mean:", scores.mean())
# print("CV std:", scores.std())


# --- Bin Tumor_Size into Small, Medium, Large ---
df['Tumor_Size_Bin'] = pd.qcut(df['Tumor_Size'], q=3, labels=['Small', 'Medium', 'Large'])

# --- Loop over each Tumor_Size bin ---
bin_models = {}
for bin_label in ['Small', 'Medium', 'Large']:
    print(f"\nTraining model for Tumor_Size bin: {bin_label}")
    
    # Subset data for this bin
    df_bin = df[df['Tumor_Size_Bin'] == bin_label].copy()
    
    # Split data into features and label
    X = df_bin.drop(columns=["Histology", "Tumor_Size_Bin"])
    y = df_bin["Histology"]
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    numeric_cols = X.select_dtypes(include=['int64','float64']).columns
    
    # Train-test split (stratify by histology)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale numerical columns
    scaler = MinMaxScaler()
    X_train_num = pd.DataFrame(scaler.fit_transform(X_train[numeric_cols]), columns=numeric_cols, index=X_train.index)
    X_test_num = pd.DataFrame(scaler.transform(X_test[numeric_cols]), columns=numeric_cols, index=X_test.index)
    
    # One-Hot Encode categorical
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    X_train_cat = pd.DataFrame(ohe.fit_transform(X_train[categorical_cols]), columns=ohe.get_feature_names_out(categorical_cols), index=X_train.index)
    X_test_cat = pd.DataFrame(ohe.transform(X_test[categorical_cols]), columns=ohe.get_feature_names_out(categorical_cols), index=X_test.index)
    
    # Combine numeric + categorical
    X_train_enc = pd.concat([X_train_num, X_train_cat], axis=1)
    X_test_enc = pd.concat([X_test_num, X_test_cat], axis=1)
    
    # Encode label
    ley = LabelEncoder()
    y_train_enc = ley.fit_transform(y_train)
    y_test_enc = ley.transform(y_test)
    
    # Train Random Forest
    random_forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42
    )
    random_forest.fit(X_train_enc, y_train_enc)
    
    # Evaluate
    train_accuracy = random_forest.score(X_train_enc, y_train_enc)
    test_accuracy = random_forest.score(X_test_enc, y_test_enc)
    
    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
    
    # Cross-validation
    scores = cross_val_score(random_forest, X_train_enc, y_train_enc, cv=5)
    print("CV mean:", scores.mean())
    print("CV std:", scores.std())
    
    # Predictions
    y_pred_enc = random_forest.predict(X_test_enc)
    y_pred = ley.inverse_transform(y_pred_enc)
    y_actual = ley.inverse_transform(y_test_enc)
    
    # Store model and encoder
    bin_models[bin_label] = {
        "model": random_forest,
        "label_encoder": ley,
        "features": X_train_enc.columns.tolist()
    }





# # Drop unnecessary columns, Survival_Rate and Follow_Up_Required are possible data leaks
# df = df.drop(columns=["Patient_ID"])

# # Split data into features and label
# X = df.drop(columns=["Tumor_Growth_Rate"])
# y = df["Tumor_Growth_Rate"]

# categorical_cols = X.select_dtypes(include=['object']).columns
# numeric_cols = X.select_dtypes(include=['int64','float64']).columns

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale numerical columns
# scaler = MinMaxScaler()
# X_train_num = pd.DataFrame(scaler.fit_transform(X_train[numeric_cols]),columns=numeric_cols,index=X_train.index)
# X_test_num = pd.DataFrame(scaler.transform(X_test[numeric_cols]),columns=numeric_cols,index=X_test.index)

# # One-Hot Encode categorical
# ohe = OneHotEncoder(sparse_output=False, drop='first')
# X_train_cat = pd.DataFrame(ohe.fit_transform(X_train[categorical_cols]),columns=ohe.get_feature_names_out(categorical_cols),index=X_train.index)
# X_test_cat = pd.DataFrame(ohe.transform(X_test[categorical_cols]),columns=ohe.get_feature_names_out(categorical_cols),index=X_test.index)

# # Combine numeric + categorical
# X_train_enc = pd.concat([X_train_num, X_train_cat], axis=1)
# X_test_enc = pd.concat([X_test_num, X_test_cat], axis=1)

# # Train Random Forest
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error

# random_forest = rf = RandomForestRegressor(random_state=42, n_jobs=-1)
# random_forest.fit(X_train_enc, y_train)

# # Predictions
# y_train_pred = random_forest.predict(X_train_enc)
# y_test_pred = random_forest.predict(X_test_enc)

# # Evaluate
# train_rmse = root_mean_squared_error(y_train, y_train_pred)
# test_rmse = root_mean_squared_error(y_test, y_test_pred)

# train_r2 = r2_score(y_train, y_train_pred)
# test_r2 = r2_score(y_test, y_test_pred)

# train_mae = mean_absolute_error(y_train, y_train_pred)
# test_mae = mean_absolute_error(y_test, y_test_pred)

# print(f"Train RMSE: {train_rmse:.3f}, R2: {train_r2:.3f}, Train MAE: {train_mae:.3f}")
# print(f"Test RMSE: {test_rmse:.3f}, R2: {test_r2:.3f}, Test MAE: {test_mae:.3f}")

# # Cross-validation
# cv_scores = cross_val_score(random_forest, X_train_enc, y_train, cv=5, scoring='neg_root_mean_squared_error')

# print("CV RMSE mean:", -np.mean(cv_scores))
# print("CV RMSE std:", np.std(cv_scores))

# # Get feature importances
# importances = random_forest.feature_importances_
# feature_importance_df = pd.DataFrame({'Feature': X_train_enc.columns,'Importance': importances}).sort_values(by='Importance', ascending=False)
# print("Feature importance:", feature_importance_df)

# # # Identify columns to keep
# # cols_to_keep = feature_importance_df.head(3)["Feature"].tolist()

# # X_train_enc = X_train_enc[cols_to_keep]
# # X_test_enc = X_test_enc[cols_to_keep]

# # print("Columns kept for model:", X_train_enc.columns.tolist())

# # # Retrain Random Forest
# # RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=5, max_features='sqrt',random_state=42)
# # random_forest.fit(X_train_enc, y_train)

# # # Predictions
# # y_train_pred = random_forest.predict(X_train_enc)
# # y_test_pred = random_forest.predict(X_test_enc)

# # # Evaluate
# # train_rmse = root_mean_squared_error(y_train, y_train_pred)
# # test_rmse = root_mean_squared_error(y_test, y_test_pred)

# # train_r2 = r2_score(y_train, y_train_pred)
# # test_r2 = r2_score(y_test, y_test_pred)

# # train_mae = mean_absolute_error(y_train, y_train_pred)
# # test_mae = mean_absolute_error(y_test, y_test_pred)

# # print(f"Train RMSE: {train_rmse:.3f}, R2: {train_r2:.3f}, Train MAE: {train_mae:.3f}")
# # print(f"Test RMSE: {test_rmse:.3f}, R2: {test_r2:.3f}, Test MAE: {test_mae:.3f}")

# # # Cross-validation
# # cv_scores = cross_val_score(random_forest, X_train_enc, y_train, cv=5, scoring='neg_root_mean_squared_error')

# # print("CV RMSE mean:", -np.mean(cv_scores))
# # print("CV RMSE std:", np.std(cv_scores))
