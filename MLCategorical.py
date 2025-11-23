import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Models
from sklearn.linear_model import LogisticRegression

# Metrics
from sklearn.metrics import accuracy_score, classification_report

# Vectorizer


data = pd.read_csv("final_dataset.csv")

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
