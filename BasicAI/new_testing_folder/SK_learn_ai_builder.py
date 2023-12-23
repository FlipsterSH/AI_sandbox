import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('new_testing_folder/test_data.csv')
x = data.drop(columns=['label']).values #data to train with per row
y = data['label'] #the right label to the data in each row

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(x.shape)
print(y.shape)


model = svm.SVC(kernel='linear')

model.fit(x_train, y_train)
predictions = model.predict(x_test)
print(x_test)

joblib.dump(model, 'new_testing_folder/ai_test_model')

for i in range(len(x_test)):
    print(f"Test: {x_test[i]}, prediction: {predictions[i]}, actual: {sum(x_test[i])}")


score = accuracy_score(y_test, predictions)
print(score)