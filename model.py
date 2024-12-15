import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

class TitanicModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder_sex = LabelEncoder()
        self.label_encoder_embarked = LabelEncoder()

    def train_model(self, url):
        data = pd.read_csv(url)
        # Data preprocessing steps here
        X = data.drop(columns=['Survived'])
        y = data['Survived']
        X = self.scaler.fit_transform(X)
        self.model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        self.model.fit(X, y)

    def predict(self, input_data):
        input_df = pd.DataFrame([input_data])
        # Preprocess input data
        input_df = self.scaler.transform(input_df)
        return self.model.predict(input_df)

