import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data():
    train_data = pd.read_csv('data/raw/train.csv')
    test_data = pd.read_csv('data/raw/test.csv')
    return train_data, test_data

def preprocess_data(train_data, test_data):
    all_data = pd.concat([train_data, test_data], sort=False)
    
    all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
    
    all_data['Age'].fillna(all_data['Age'].median(), inplace=True)
    all_data['Fare'].fillna(all_data['Fare'].median(), inplace=True)
    all_data['Embarked'].fillna(all_data['Embarked'].mode()[0], inplace=True)
    
    categorical_features = ['Sex', 'Embarked', 'Pclass']
    numeric_features = ['Age', 'Fare', 'FamilySize']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
        ])
    
    X = all_data[numeric_features + categorical_features]
    y = all_data['Survived']
    
    return X, y, preprocessor

if __name__ == "__main__":
    train_data, test_data = load_data()
    X, y, preprocessor = preprocess_data(train_data, test_data)
    print("Data preprocessing completed.")
