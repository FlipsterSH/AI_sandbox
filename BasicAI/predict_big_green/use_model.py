import joblib
import yfinance as yf



def percent_numb(last, now): #last is the price yesterday, now is the price today(most recent price)
    prosent = 0
    if last <= now:
        prosent = ((now / last) - 1) * 100

    if last > now:
        prosent = (1 - (now / last)) * -100

    return round(prosent) #returns the percentage difference number between the two prices



def get_stock_data_interval(ticker, interval): #returns a list of percent changes for the stock in the most recent interval
    data = yf.download(ticker, period="1mo", interval="1d") 
    stock_prices = data['Close'] #closing prices of the stock
    interval_prices = stock_prices[-interval - 1:len(stock_prices)]

    stock_percent_changes = [] 
    for i in range(len(interval_prices) - 1):
        stock_percent_changes.append(percent_numb(interval_prices[i], interval_prices[i + 1]))

    print("Data for prediction:", stock_percent_changes)
    return stock_percent_changes #returns a list of price percent changes for the most recent interval


def use_model(model_file, data): #takes in a model and a list of percent changes, returns a prediction for the next day
    model = joblib.load(model_file)
    prediction = model.predict([data])
    if prediction == [1]:
        print("Prediction: Market will go up or be flat")
    else:
        print("Prediction: Market will go down")




if __name__ == "__main__":
    data = get_stock_data_interval("^GSPC", 10) # <--- Here you can specify the time interval you want to use for the prediction
    use_model('market_move_predicter.joblib', data)
    

