import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Numeric machine learning model with RandomForestClassifier and LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Metrics
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance

# Create a dataframe of the csv file
data = pd.read_csv("final_dataset.csv")

x = data[["price","overall rating","number sold","total review"]]
y = data["sentiment"]

original_features = ["price", "overall rating", "number sold", "total review"]

print("x shape:", x.shape)

# Use standard scaler before adding polynomial features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x_scaled)

feature_names_poly = poly.get_feature_names_out(original_features)

print("x shape after polynomial:", x_poly.shape)

# Split the data into training and testing sets
x_train_poly, x_test_poly, y_train, y_test = train_test_split(
    x_poly, 
    y, 
    test_size=0.2,
    random_state=42
)

# Another data split for the x data with no added polynomial features (this is for the tree based models)
x_train, x_test, _, _ = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

print("x_train_poly and x_test_poly shape: ", x_train_poly.shape, x_test_poly.shape)
print("x_train and x_test shape: ", x_train.shape, x_test.shape)

LR = LogisticRegression(max_iter=1000)
HGB = HistGradientBoostingClassifier()
RF = RandomForestClassifier()

# MODEL TRAINING
print("Logistic Regression (with polynomial features)")
LR.fit(x_train_poly, y_train)
y_pred_lr = LR.predict(x_test_poly)
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

print("\nHist Gradient Boosting Classifier")
HGB.fit(x_train, y_train)
y_pred_hgb = HGB.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred_hgb))
print(classification_report(y_test, y_pred_hgb))

print("\nRandom Forest Classifier")
RF.fit(x_train, y_train)
y_pred_rf = RF.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


'''
# FEATURE IMPORTANCES
# Logistic regression 
print("\nLogistic Regression feature importance (from Polynomial Features):")
lr_coef = np.abs(LR.coef_[0])  # abs because direction doesn't matter
collapsed = {feat: 0 for feat in original_features}
for name, coef in zip(feature_names_poly, lr_coef):
    for orig in original_features:
        if orig in name:
            collapsed[orig] += coef
for feat, val in sorted(collapsed.items(), key=lambda x: x[1], reverse=True):
    print(f"{feat}: {val:.6f}")

# Hist Gradient Boosting
print("\nHistGradientBoosting permutation importance:")
perm = permutation_importance(HGB, x_test, y_test, n_repeats=10, random_state=42)
for feat, val in sorted(zip(original_features, perm.importances_mean), key=lambda x: x[1], reverse=True):
    print(f"{feat}: {val:.6f}")

# Random forest classifier
print("\nRandom Forest Feature Importance:")
rf_importances = RF.feature_importances_
for feat, val in sorted(zip(original_features, rf_importances), key=lambda x: x[1], reverse=True):
    print(f"{feat}: {val:.6f}")


# PLOTTING
# -------- 1. Logistic Regression (collapsed) --------
plt.figure(figsize=(7, 5))
plt.barh(list(collapsed.keys()), list(collapsed.values()))
plt.xlabel("Importance (summed abs coefficients)")
plt.title("Logistic Regression (with polynomial features)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# -------- 2. HistGradientBoosting permutation importance --------
hgb_scores = perm.importances_mean
plt.figure(figsize=(7, 5))
plt.barh(original_features, hgb_scores)
plt.xlabel("Permutation Importance")
plt.title("HistGradientBoosting Permutation Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# -------- 3. RandomForest Feature Importance --------
plt.figure(figsize=(7, 5))
plt.barh(original_features, rf_importances)
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
'''

# STILL REQUIRES IMPROVEMENTSS
def recommended_price_range(model, df, price_col="price", threshold=0.7, steps=200, model_name="Model"):
    # Use a representative item (median values)
    base_row = df.median(numeric_only=True).to_frame().T

    price_min = df[price_col].min()
    price_max = df[price_col].max()
    price_grid = np.linspace(price_min, price_max, steps)

    probabilities = []

    for p in price_grid:
        row = base_row.copy()
        row[price_col] = p

        # Predict sentiment probability
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(row)[:, 1][0]
        else:
            prob = model.predict(row)[0]

        probabilities.append(prob)

    probabilities = np.array(probabilities)

    # Identify all price points that meet the threshold
    good_prices = price_grid[probabilities >= threshold]

    if len(good_prices) == 0:
        rec_min, rec_max = None, None
    else:
        rec_min = float(good_prices.min())
        rec_max = float(good_prices.max())

    # ===== PLOT =====
    plt.figure(figsize=(8, 5))
    plt.plot(price_grid, probabilities, linewidth=2)
    plt.axhline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold}")
    plt.title(f"Predicted Positive Sentiment vs Price ({model_name})")
    plt.xlabel("Price")
    plt.ylabel("Predicted Probability of Positive Sentiment")
    plt.legend()
    plt.grid(True)
    plt.show()

    return rec_min, rec_max

# ========================================================
# RUN PRICE TESTING FOR EACH MODEL
# ========================================================

# Logistic Regression (uses polynomial features)
lr_min, lr_max = recommended_price_range(
    LR, pd.DataFrame(x_test_poly, columns=feature_names_poly), 
    price_col="price", threshold=0.7, 
    model_name="Logistic Regression (Polynomial Features)"
)
print("Logistic Regression:", lr_min, lr_max)

# HistGradientBoosting (raw features)
hgb_min, hgb_max = recommended_price_range(
    HGB, x_test, price_col="price", threshold=0.7, 
    model_name="HistGradientBoosting"
)
print("HistGradientBoosting:", hgb_min, hgb_max)

# Random Forest (raw features)
rf_min, rf_max = recommended_price_range(
    RF, x_test, price_col="price", threshold=0.7, 
    model_name="Random Forest"
)
print("Random Forest:", rf_min, rf_max)

