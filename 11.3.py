import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr

# Setup
wr.filterwarnings('ignore')
sns.set_style("whitegrid")

# Load the dataset
df = pd.read_csv("eswar.csv")

# Initial view
print("First 5 rows:\n", df.head())
print("\nInfo:\n")
print(df.info())
print("\nShape of dataset:", df.shape)

# Handle missing and duplicate values
print("\nMissing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
df.fillna(0, inplace=True)

# Summary statistics
print("\nSummary statistics:\n", round(df.describe(include='all'), 1))
print("\nUnique values per column:\n", df.nunique())



# Rename columns
df.rename(columns={
    'Arrival_Date': 'Date Recorded',
    'Min_x0020_Price': 'Min Price',
    'Max_x0020_Price': 'Max Price'
}, inplace=True)

# Convert Date
df['Date Recorded'] = pd.to_datetime(df['Date Recorded'], errors='coerce')
df['Year'] = df['Date Recorded'].dt.year
df['Month'] = df['Date Recorded'].dt.month

# Calculate Average Price and Profit (Max - Min)
df['Average Price'] = (df['Min Price'] + df['Max Price']) / 2
df['Profit'] = df['Max Price'] - df['Min Price']

# Top 5 Expensive Markets (by average price)
print("\nTop 5 Markets by Average Price:")
print(df.groupby('Market')['Average Price'].mean().sort_values(ascending=False).head(5))

# Plot 1: Average Price by Commodity
plt.figure(figsize=(12,6))
sns.barplot(data=df, x='State', y='Average Price', estimator=np.mean, ci=None, errorbar=None)
plt.xticks(rotation=90, ha='right')
plt.title('Average Price by Commodity')
plt.tight_layout()
plt.show()


# Plot 2: Pie Chart - Distribution by State
state_counts = df['State'].value_counts()
plt.figure(figsize=(8,8))
plt.pie(state_counts, labels=state_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Entries by State')
plt.axis('equal')
plt.show()

# Plot 3: Scatter - Min vs Max Price colored by Grade
if 'Grade' in df.columns:
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x='Min Price', y='Max Price', hue='Grade')
    plt.title("Min vs Max Price by Grade")
    plt.tight_layout()
    plt.show()


# Plot 5: Correlation Heatmap
plt.figure(figsize=(6,5))
sns.heatmap(df[['Min Price', 'Max Price', 'Average Price', 'Profit']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# Plot 6: Boxplot by Variety
plt.figure(figsize=(12,6))
sns.boxplot(data=df, x='Grade', y='State')
plt.xticks(rotation=45, ha='right')
plt.title('Price Distribution by Variety')
plt.tight_layout()
plt.show()

# Plot 7: Histogram of Profit
plt.figure(figsize=(10,6))
sns.histplot(df['Profit'], bins=20, kde=True, color='green')
plt.title("Profit Distribution (Max - Min Price)")
plt.xlabel("Profit")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

