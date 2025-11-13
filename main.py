import pandas as pd
import numpy as np

df = pd.read_csv('brain_tumor_dataset.csv')

print(df.describe())
print(df.info())

# most common tumor types for male and female? histogram of all tumor types for everyon
# histogram of average size of each tumor type