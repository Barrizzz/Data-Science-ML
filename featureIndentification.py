import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

df = pd.read_csv("merged_dataset.csv")

# List of potential features
# Numerical features
numericalF = [
    "price",
    "overall rating",
    "number sold",
    "total review",
    "customer rating",
    "salary"
]

# Categorical features
categoricalF = [
    "category",
    "location",
    "region",
    "emotion"
]

# Other features
otherF = [
    "product name",
    "customer review"
]

# LABEL
label = "sentiment"

# Spearmann corellation using pandas
print("Spearman correlation of the numerical features:")
spearman_corr = df[numericalF + [label]].corr(method="spearman")
print(spearman_corr)

# Cramer's V
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

print("\nCramer's V correlation of the categorical features")
for i in categoricalF:
    cramersv_corr = cramers_v(df[i], df[label])
    print(f"{i}: {cramersv_corr}")


# Determining features to drop from the dataset
features_to_drop = []

# Check numerical
for i in numericalF:
    if abs(spearman_corr.loc[i, label]) > 0.85:
        features_to_drop.append(i)

for i in categoricalF:
    if cramers_v(df[i], df[label]) > 0.85:
        features_to_drop.append(i)

print("\nFeatures to drop:", features_to_drop)

# Taking the insight from our correlation analysis, we have seen that the feature 'customer rating' 
# in the numerical features have a very high correlation with the label, this could lead to overfitting, and the machine learning
# model will ignore the other important features. Therefore, we should drop this column.
# We also decided to drop the emotion because it also has a high correlation with our sentiment label.

df = df.drop(columns=["customer rating", "emotion"])
df.to_csv("final_dataset.csv", index=False)