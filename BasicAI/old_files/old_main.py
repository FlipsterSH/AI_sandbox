import yfinance as yf
import requests
import csv
import requests
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import joblib
import pandas as pd

GREEN = 1
RED = 0


def percent_numb(last, now): #last is the price yesterday, now is the price today(most recent price)
    prosent = 0
    if last <= now:
        prosent = ((now / last) - 1) * 100

    if last > now:
        prosent = (1 - (now / last)) * -100

    return round(prosent) #returns the percentage difference number between the two prices

def color_change(last, now): #last is the price yesterday, now is the price today(most recent price)
    if last <= now:
        return 1 #green
    else:
        return 0 #red


def get_sp500_data(): #returns a dictionary of percent changes and color changes for the SP500
    data = yf.download("^GSPC", period="max")
    sp500_prices = data['Close'] #closing prices of the SP500

    sp500_percent_changes = [] #percent price changes for closing price each day
    for i in range(len(sp500_prices) - 1):
        sp500_percent_changes.append(percent_numb(sp500_prices[i], sp500_prices[i + 1]))

    sp500_color_changes = [] #color changes for closing price each day
    for i in range(len(sp500_prices) - 1):
        sp500_color_changes.append(color_change(sp500_prices[i], sp500_prices[i + 1]))

    sp500_data = {}
    sp500_data['percent_changes'] = sp500_percent_changes
    sp500_data['color_changes'] = sp500_color_changes
    
    return sp500_data #returns a dictionary of price percent changes and color changes(if a day is green or red)



def prepare_color_data_csv(data): #takes in dicitonary of data, containing percent changes and color changes
    with open('data.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['P1', 'P2','P3', 'P4', 'P5', 'NEXT_COLOR'])

        for i in range(len(data['percent_changes']) - 5):
            row = []
            for j in range(6):
                if j == 5:
                    if data['percent_changes'][i + j] >= 0:
                        row.append(GREEN)
                    else:
                        row.append(RED)
                
                else:
                    row.append(data['percent_changes'][i + j])
                
            # for j in range(6):    
            #     row.append(data['color_changes'][i + j])

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
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    #joblib.dump(model, 'PE_ratio_predicter.joblib')

    score = accuracy_score(y_test, predictions)
    print(score)


if __name__ == "__main__":
    data = get_sp500_data()
    prepare_color_data_csv(data)
    
    print('Processing data...')
    create_model('data.csv')