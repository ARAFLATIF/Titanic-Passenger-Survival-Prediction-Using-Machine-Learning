from flask import Flask, render_template, request, jsonify
from model import TitanicModel

app = Flask(__name__)

model = TitanicModel()
model.train_model('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
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
    return jsonify({'prediction': survival})

if __name__ == '__main__':
    app.run(debug=True)
