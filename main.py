from flask import Flask, render_template, request
from model import TitanicModel
import os

app = Flask(__name__)

model = TitanicModel()
model.train_model('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = {
            'Pclass': int(request.form['Pclass']),
            'Sex': request.form['Sex'],
            'Age': float(request.form['Age']),
            'Fare': float(request.form['Fare']),
            'Embarked': request.form['Embarked'],
            'FamilySize': int(request.form['FamilySize'])
        }
        prediction = model.predict(features)
        survival = 'Survived' if prediction[0] == 1 else 'Did Not Survive'
        return render_template('index.html', prediction=survival)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

