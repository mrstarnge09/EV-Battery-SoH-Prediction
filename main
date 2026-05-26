# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Dataset
df = pd.read_csv('ev.csv')

# 3. Basic Checks
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

# 4. Check for missing values
print(df.isnull().sum())

# 5. Visualize important columns
# Example plots you should make:
plt.figure(figsize=(10,6))
sns.histplot(df['battery_health_%'])   # or 'Capacity' or 'SoH'
plt.title('Distribution of Battery Health')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
