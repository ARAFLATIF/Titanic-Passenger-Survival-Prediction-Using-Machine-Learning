import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

class TitanicModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder_sex = LabelEncoder()
        self.label_encoder_embarked = LabelEncoder()

    def train_model(self, url):
        data = pd.read_csv(url)
        data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
        data['Age'].fillna(data['Age'].median(), inplace=True)
        data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
        data['Sex'] = self.label_encoder_sex.fit_transform(data['Sex'])
        data['Embarked'] = self.label_encoder_embarked.fit_transform(data['Embarked'])
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
        data.drop(columns=['SibSp', 'Parch'], inplace=True)

        X = data.drop(columns=['Survived'])
        y = data['Survived']
        X = self.scaler.fit_transform(X)

        self.model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        self.model.fit(X, y)

    def predict(self, input_data):
        input_df = pd.DataFrame([input_data])
        input_df['Sex'] = self.label_encoder_sex.transform(input_df['Sex'])
        input_df['Embarked'] = self.label_encoder_embarked.transform(input_df['Embarked'])
        input_df = self.scaler.transform(input_df)
        return self.model.predict(input_df)
