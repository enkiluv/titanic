# -*- coding: utf-8 -*-

# Import necessary libraries
import pandas as pd
import ai_wonder as wonder

# Input with default values
def user_input(prompt, default):
    response = input(f"{prompt} (default: {default}): ")
    return response if response else default

# The driver
if __name__ == "__main__":
    print(f"Titanic 'Survived' Predictor")
    print("Powered by AI Wonder\n")
    
    # User inputs
    Sex = user_input("Sex", "'male'")
    Embarked = user_input("Embarked", "'Q'")
    Pclass = int(user_input("Pclass", 1))
    Age = float(user_input("Age", 34.0))
    SibSp = int(user_input("SibSp", 0))
    Parch = int(user_input("Parch", 0))
    Fare = float(user_input("Fare", 26.55))

    # Make datapoint from user input
    point = pd.DataFrame([{
        'Sex': Sex,
        'Embarked': Embarked,
        'Pclass': Pclass,
        'Age': Age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Fare': Fare,
    }])

    # Predict
    model = wonder.load_model('titanic_model.pkl')
    prediction = str(model.predict(point)[0])
    print(f"\nPrediction of 'Survived' is '{prediction}'.")
###
