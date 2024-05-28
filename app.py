from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load(f"C:\Users\Gaurav\OneDrive\Desktop\codsoft2\Model_train\logistic.pkl")

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the request
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Make prediction
    prediction = model.predict(np.array([[sepal_length, sepal_width, petal_length, petal_width]]))

    # Map prediction index to Iris species
    species_mapping = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    predicted_species = species_mapping[prediction[0]]

    return render_template('index.html', prediction_text='Predicted Iris Species: {}'.format(predicted_species))

if __name__ == '__main__':
    app.run(debug=True)
