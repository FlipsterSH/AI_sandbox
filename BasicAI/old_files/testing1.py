import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

color_data = pd.read_csv('old_files/testing.csv')
X = color_data.drop(columns=['fav_color']).values
y = color_data['fav_color']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X.shape)
print(y.shape)


#model = joblib.load('favourite_color_predictor.joblib')

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

joblib.dump(model, 'favourite_color_predictor.joblib')

print(y_test)
print(predictions)

score = accuracy_score(y_test, predictions)
print(score)