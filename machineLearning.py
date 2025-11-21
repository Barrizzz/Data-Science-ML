import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

# Text analysis classification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Create a dataframe of the csv file
data = pd.read_csv("cleaned_data_tokopedia.csv")

'''
x = data.drop(columns=["sentiment", "category", "product name", "location", "customer review", "emotion"], axis=1)
y = data["sentiment"]

print("x shape: ", x.shape)
poly = PolynomialFeatures()
x = poly.fit_transform(x)
print("x shape after polynomial features: ", x.shape)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size=0.2,
    random_state=42
)

print("x_train and x_test shape: ", x_train.shape, x_test.shape)


LR = LinearRegression()
HGB = HistGradientBoostingClassifier()
RF = RandomForestClassifier()
models = [LR, HGB, RF]
models[2].fit(x_train, y_train)


# Predicting the target by usng x_test, then we compare it with y_test
y_pred = models[2].predict(x_test)
r2 = r2_score(y_test, y_pred)
print(models[2], r2)


# price, overall rating (average customer rating), total sold, total reviews, customer rating
example_data = [1000000, 4.9, 10, 100, 4]
example_data_poly = poly.transform([example_data])
example_pred = models[2].predict(example_data_poly)

print("Example data prediction: ", example_pred)
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

# This is to convert the sliced customer review data from the dataframe into numpy array
#customer_reviews = data.iloc[:4000:20]['customer review'].to_numpy()
customer_reviews = [
    'kalungnya bagus banget woi',
    'kalungnya jelek banget sumpah jadi gatel terus baru sekali pake langsung karatan'
]

#print(customer_reviews)

for review in customer_reviews:
    example_reviews_vec = vectorizer.transform([review])
    example_pred = models[0].predict(example_reviews_vec)
    if example_pred[0] == 1:
        print("GOOD")
    else:
        print("BAD")
