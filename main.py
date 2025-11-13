import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('brain_tumor_dataset.csv')

print(df.head())
print(df.describe())
print(df.info())

# most common tumor types for male and female? histogram of all tumor types for everyon
# histogram of average size of each tumor type

plt.figure(figsize=(10,6))
sns.countplot(data=df, x='Tumor Type', hue='Gender', 
              order=df['Tumor Type'].value_counts().index)
plt.title('Most Common Tumor Types by Gender', fontsize=14)
plt.xlabel('Tumor Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Gender')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(data=df, x='Tumor Type', 
              order=df['Tumor Type'].value_counts().index, 
              color='skyblue')

plt.title('Distribution of Tumor Types (All Patients)', fontsize=14)
plt.xlabel('Tumor Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Compute average tumor size per tumor type
avg_size = df.groupby('Tumor Type')['Size (cm)'].mean().sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=avg_size.index, y=avg_size.values, palette='coolwarm')
plt.title('Average Tumor Size by Tumor Type', fontsize=14)
plt.xlabel('Tumor Type')
plt.ylabel('Average Size (cm)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
