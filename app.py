# from pyparsing import null_debug_action
from flask import Flask, render_template, request

import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#import numpy as np
app = Flask(__name__)

d = pd.read_csv(
    '/Users/divyankshah/Documents/miniproject/Crop_recommendation.csv')
output = {
    "rice": 1,
    "maize": 2,
    "chickpea": 3,
    "kidneybeans": 4,
    "pigeonpeas": 5,
    "mothbeans": 6,
    "mungbean": 7,
    "blackgram": 8,
    "lentil": 9,
    "pomegranate": 10,
    "banana": 11,
    "mango": 12,
    "grapes": 13,
    "watermelon": 14,
    "muskmelon": 15,
    "apple": 16,
    "orange": 17,
    "papaya": 18,
    "coconut": 19,
    "cotton": 20,
    "jute": 21,
    "coffee": 22
}
d = d.drop("label", axis=1)
scalerX = MinMaxScaler()
print(scalerX.fit(d))


def crop_name(val):
    for key, value in output.items():
        if val == value:
            return key

    return "key doesn't exist"


filename = 'Crop_Recommendation.pkl'
loaded_model = pickle.load(open(filename, 'rb'))


@app.route("/")
def hello():
    return render_template("index.html")


@app.route("/sub", methods=["POST"])
def submit():
    if request.method == "POST":
        n = request.form["nitrogen"]
        p = request.form["phosphorus"]
        k = request.form["potassium"]
        t = request.form["temperature"]
        h = request.form["humidity"]
        ph = request.form["ph"]
        r = request.form["rainfall"]

        a = np.array([n, p, k, t, h, ph, r])
        a = a.reshape(1, 7)

        a = scalerX.transform(a)
        a = np.array(a)

        prediction = loaded_model.model.predict(a)

        answer = prediction.argmax()
    
        print("Crop you must grow is : ", crop_name(answer))
        img = "static/images/"+crop_name(answer)+".jpg"
    return render_template("submit.html", crop=crop_name(answer), temp=img)


if __name__ == "__main__":
    app.run(debug=True)
