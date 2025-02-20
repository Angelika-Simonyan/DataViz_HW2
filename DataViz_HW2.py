import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde


cancer_data = pd.read_csv("C:/Users/HP/OneDrive/Рабочий стол/DataViz/HW2/lung_cancer_prediction_dataset.csv")
pollution_data = pd.read_csv("C:/Users/HP/OneDrive/Рабочий стол/DataViz/HW2/global_air_pollution_dataset.csv")

# Part 3.1
column_name = "Annual_Lung_Cancer_Deaths"
plt.figure(figsize=(8, 5))
sns.boxplot(y=cancer_data[column_name], boxprops=dict(facecolor="deeppink"))
plt.title("Distribution of Lung Cancer Deaths")
plt.ylabel("Number of Deaths")
# plt.show()

# Part 3.2
plt.figure(figsize=(8, 5))
sns.histplot(pollution_data["PM2.5_AQI_Value"], bins=30, kde=True, color="deeppink")
plt.title("Histogram of PM2.5 AQI Values")
plt.xlabel("PM2.5 AQI Value")
plt.ylabel("Frequency")
# plt.show()

# Part 3.3
plt.figure(figsize=(8, 5))
sns.kdeplot(cancer_data["Mortality_Rate"], fill=True, color="deeppink")
plt.title("Density Plot of Lung Cancer Mortality Rate")
plt.xlabel("Mortality Rate")
plt.ylabel("Density")
# plt.show()


# Part 4.1
pm25_values = pollution_data["PM2.5_AQI_Value"].dropna()
plt.figure(figsize=(8, 5))
counts, bins, _ = plt.hist(pm25_values, bins=30, density=True, alpha=0.6, color="blue", edgecolor="black", label="Histogram")

# Density plot
kde = gaussian_kde(pm25_values)
x_vals = np.linspace(min(pm25_values), max(pm25_values), 1000)
plt.plot(x_vals, kde(x_vals), color="darkred", linewidth=2, label="Density Plot (KDE)")
plt.fill_between(x_vals, kde(x_vals), color="darkred", alpha=0.3)

plt.title("PM2.5 AQI Distribution with Density Overlay", fontsize=12, fontweight="bold")
plt.xlabel("PM2.5 AQI Value")
plt.ylabel("Density")
plt.legend()
plt.figtext(0.1, 0.01, "This plot represents the distribution of PM2.5 AQI values with a density overlay.", wrap=True, fontsize=8, ha="center")
plt.subplots_adjust(bottom=0.2)
# I could not get the description with 1 line, no matter how many adjustments I made

# plt.show()
