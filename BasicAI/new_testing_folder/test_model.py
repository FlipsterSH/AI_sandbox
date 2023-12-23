import joblib



def use_model(model_file, data): #takes in a model and a list of percent changes, returns a prediction for the next day
    model = joblib.load(model_file)
    prediction = model.predict([data])
    print(prediction)



if __name__ == "__main__":
    data = [6,6,6]
    use_model('new_testing_folder/ai_test_model', data)