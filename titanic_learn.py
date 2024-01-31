# -*- coding: utf-8 -*-

# Import Packages
import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier

def load_data(filepath):
    # Load dataset
    data = pd.read_csv(filepath, skipinitialspace=True)
    # Remove any 'Unnamed:' columns
    data = data.loc[:, ~data.columns.str.startswith('Unnamed: ')]
    # Drop uninterested columns
    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    return data

def preprocess_data(data, target_col):
    # Split X and y
    X = data.drop(target_col, axis=1)
    y = data[target_col].values

    # Encode categorical target
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y.astype(str).ravel())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Classify column types
    cat_columns = X_train.select_dtypes(include='object').columns
    num_columns = X_train.select_dtypes(exclude='object').columns

    # Impute missing values
    X_train[cat_columns] = X_train[cat_columns].fillna("<NA>")
    X_test[cat_columns] = X_test[cat_columns].fillna("<NA>")
    medians = X_train[num_columns].median()
    X_train[num_columns] = X_train[num_columns].fillna(medians)
    X_test[num_columns] = X_test[num_columns].fillna(medians)

    # Encode categorical columns
    encoder = preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1)
    train_cat_values = encoder.fit_transform(X_train[cat_columns])
    test_cat_values = encoder.transform(X_test[cat_columns])

    # Scale numeric columns
    scaler = preprocessing.MinMaxScaler()
    train_num_values = scaler.fit_transform(X_train[num_columns])
    test_num_values = scaler.transform(X_test[num_columns])

    # Rebuild train and test dataset
    X_train = np.hstack((train_cat_values, train_num_values))
    X_test = np.hstack((test_cat_values, test_num_values))

    return X_train, X_test, y_train, y_test, label_encoder

def train_model(X_train, y_train):
    # Build a model with best possible parameters
    model = LGBMClassifier(**{
        "n_estimators": 4,
        "num_leaves": 4,
        "min_child_samples": 10,
        "learning_rate": 1.0,
        "log_max_bin": 8,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.02,
        "reg_lambda": 0.03
    })

    # And train (fit) it
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test, label_encoder):
    # Predict using the test dataset
    y_test = label_encoder.inverse_transform(y_test)
    y_pred = label_encoder.inverse_transform(model.predict(X_test))
    y_yhat = pd.DataFrame({'Real': y_test, 'Pred': y_pred})
    print(y_yhat.reset_index(drop=True).round(2), '\n')

    # Calculate performance metrics
    print("Classification Report")
    print(classification_report(y_test, y_pred), '\n')

def main():
    # Load and explore data
    data = load_data("titanic.csv")
    print('Dataset Shape:', data.shape, '\n')
    print(data.head().round(2), '\n')
    data.info()
    print()
    print(data.describe(), '\n')

    # Preprocess data
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(data, 'Survived')

    # Train and evaluate model
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, label_encoder)

if __name__ == "__main__":
    main()
