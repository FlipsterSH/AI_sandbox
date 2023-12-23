import yfinance as yf
import csv
import requests
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import joblib
import pandas as pd

GREEN = 1
RED = -1


def percent_numb(last, now): #last is the price yesterday, now is the price today(most recent price)
    prosent = 0
    if last <= now:
        prosent = ((now / last) - 1) * 100

    if last > now:
        prosent = (1 - (now / last)) * -100

    return round(prosent, 1) #returns the percentage difference number between the two prices


def get_stock_data(): #returns a dictionary of percent changes and color changes for the stock
    data = yf.download("^GSPC", period="max", interval="1d") #^GSPC is the S&P 500 index
    stock_prices = data['Close'] #closing prices of the stock

    stock_percent_changes = [] #percent price changes for closing price each day
    for i in range(len(stock_prices) - 1):
        stock_percent_changes.append(percent_numb(stock_prices[i], stock_prices[i + 1]))
    
    return stock_percent_changes #returns a list of percent changes for the stock



def prepare_data_csv(data, interval): #takes in dicitonary of data, containing percent changes and time interval
    with open('predict_big_green/data.csv', 'w') as f:
        writer = csv.writer(f)
        
        #print(data)

        first_row = [] #creating the label for the first row of the csv file
        for i in range(interval):
            first_row.append('P' + str(i + 1))
        first_row.append('NEXT_COLOR')
        writer.writerow(first_row) #NEXT_COLOR is the color of the next day, if the price goes up or down

        for i in range(len(data) - interval): #creating the data for the csv file, used for training the model
            row = []
            for j in range(interval + 1):
                #print(data[i + j])
                if j == interval:
                    if round(data[i + j]) <= 0:
                        row.append(RED)
                    else:
                        row.append(GREEN)
                else:
                    row.append(round(data[i + j]))

                #print(row)

            writer.writerow(row)
    print("data is ready")



def create_model(csv_file):
    color_data = pd.read_csv(csv_file)
    X = color_data.drop(columns=['NEXT_COLOR']).values
    y = color_data['NEXT_COLOR']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # label_encoder = preprocessing.LabelEncoder()
    # encoded_y = label_encoder.fit_transform(y_train)
    # encoded_y_test = label_encoder.fit_transform(y_test)

    #model = DecisionTreeClassifier()
    model = svm.SVC(kernel='rbf')
    #model = RandomForestClassifier(n_estimators=100)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    joblib.dump(model, 'predict_big_green/predict_big_green.joblib')

    score = accuracy_score(y_test, predictions)
    print(score)


if __name__ == "__main__":
    data = get_stock_data()
    prepare_data_csv(data, 2)
    
    print('Processing data...')
    create_model('predict_big_green/data.csv')