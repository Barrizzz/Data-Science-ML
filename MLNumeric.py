import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, stdev

# Numeric machine learning model with RandomForestClassifier and LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Metrics
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance

# Create a dataframe of the csv file
data = pd.read_csv("Data-Science-ML/final_dataset.csv")

x = data[["price","overall rating","number sold","total review"]]
y = data["sentiment"]

original_features = ["price", "overall rating", "number sold", "total review"]

print("x shape:", x.shape)

# Use standard scaler for logistic regression
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Another data split for the x data with no added polynomial features (this is for the tree based models)
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

<<<<<<< HEAD
=======
print("x_train_poly and x_test_poly shape: ", x_train_poly.shape, x_test_poly.shape)
print("x_train and x_test shape: ", x_train.shape, x_test.shape)

>>>>>>> ac10936373f5016df56edfb1bd6c33fa496e3e38
LR = LogisticRegression()
HGB = HistGradientBoostingClassifier()
RF = RandomForestClassifier()

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

def test_model(model):
    lst_accu_stratified = []
    for train_index, test_index in skf.split(x, y):
        if isinstance(model, LogisticRegression):
            x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index]
        else:
            x_train_fold, x_test_fold = x.iloc[train_index], x.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        model.fit(x_train_fold, y_train_fold)
        lst_accu_stratified.append(model.score(x_test_fold, y_test_fold))
        
    print('List of possible accuracy:', lst_accu_stratified)
    print('\nMaximum Accuracy That can be obtained from this model is:',
        max(lst_accu_stratified))
    print('\nMinimum Accuracy:',
        min(lst_accu_stratified))
    print('\nOverall Accuracy:',
        mean(lst_accu_stratified))
    print('\nStandard Deviation is:', stdev(lst_accu_stratified))

# TESTING HIGHEST POSSIBLE ACCURACY (not yet training the model)
# 1. Logistic Regression
test_model(LR)

# 2. Random Forest Classifier
test_model(RF)

# 3. HistGradientBoostingClassifier
test_model(HGB)


# TRAINING THE MODELS
# 1. Logistic Regression
LR.fit(x_scaled, y_train)

# 2. Random Forest Classifier
RF.fit(x_train, y_train)

# 3. HistGradientBoostingClassifier
HGB.fit(x_train, y_train)

# PREDICTIONS
pred_lr  = LR.predict(scaler.transform(x_test))
pred_rf  = RF.predict(x_test)
pred_hgb = HGB.predict(x_test)

print("LR final accuracy:", accuracy_score(y_test, pred_lr))
print("RF final accuracy:", accuracy_score(y_test, pred_rf))
print("HGB final accuracy:", accuracy_score(y_test, pred_hgb))


# CONFUSION MATRIX
# Logistic Regression confusion matrix
ConfusionMatrixDisplay.from_estimator(LR, x_scaled, y_test, cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# Random Forest Classifier confusion matrix
ConfusionMatrixDisplay.from_estimator(RF, x_test, y_test, cmap='Oranges')
plt.title("Confusion Matrix - Random Forest Classifier")
plt.show()

# HistGradientBoostingClassifier confusion matrix
ConfusionMatrixDisplay.from_estimator(HGB, x_test, y_test, cmap='Greens')
plt.title("Confusion Matrix - HistGradientBoostingClassifier")
plt.show()




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
'''
