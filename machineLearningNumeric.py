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
x = data["customer review"]
y = data["sentiment"]
# for sentiment, 1 = positive, 0 = negative

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size=0.2,
    random_state=42
)

# Vectorize the text data
vectorizer = TfidfVectorizer()
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

models = [LogisticRegression(), MultinomialNB()]

models[0].fit(x_train_vec, y_train)
y_pred = models[0].predict(x_test_vec)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))






# Inserting the reviews section
# This is to convert the sliced customer review data from the dataframe into numpy array
customer_reviews = data.iloc[:4000:20]['customer review'].to_numpy()

# Manual input customer reviews
#customer_reviews = []

#print(customer_reviews)

# Loop trough the list and use the machine learning model to predict the result
for review in customer_reviews:
    example_reviews_vec = vectorizer.transform([review])
    example_pred = models[0].predict(example_reviews_vec)
    if example_pred[0] == 1:
        print("GOOD")
    else:
        print("BAD")
'''