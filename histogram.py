import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Dataset.csv")

print(df.head())

print(df("CreditScore").mean())

print(df.columns)

mean_value = np.mean(df['CreditScore'])

median_value = np.median(df['CreditScore'])

mode_value = df['CreditScore'].mode()[0]


print("Mean    :", mean_value)
print("Median  :", median_value)
print("Mode    :", mode_value)

#histrogram

plt.figure(figsize=(10,7))
plt.hist(df["CreditScore"], bins=20,
         color="skyblue", edgecolor ="black")


plt.axvline(mean_value, color="red", linestyle="--", linewidth=2, label=f"Mean = {mean_value:.2f}")
plt.axvline(median_value, color="green", linestyle="-.", linewidth=2, label=f"Median = {median_value:.2f}")
plt.axvline(mode_value, color="orange", linestyle=":", linewidth=2, label=f"Mode = {mode_value}")plt.title(f"Histogram of {'CreditScore'}")
plt.xlabel('CreditScore')
plt.ylabel("Frequency")
plt.legend()
plt.show()


numeric_df = df.select_dtypes(include=np.number)
corr_matrix = numeric_df.corr()

# Print Correlation Matrix
print("Correlation Matrix:")
print(corr_matrix)

# Plot Heatmap
plt.figure(figsize=(8,6))
plt.imshow(corr_matrix, cmap="coolwarm", interpolation="nearest")

# Add color bar
plt.colorbar(label="Correlation")

# Axis labels
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)

plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()
