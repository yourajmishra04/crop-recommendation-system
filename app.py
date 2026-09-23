from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and scalerspip in
model = joblib.load('model.pkl')
sc = joblib.load('standscaler.pkl')
ms = joblib.load('minmaxscaler.pkl')

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        feature_list = [N, P, K, temp, humidity, ph, rainfall]

        columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        single_pred = pd.DataFrame([feature_list], columns=columns)

        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)

        prediction = model.predict(final_features)

        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
            8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
            19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }

        crop = crop_dict.get(prediction[0], "Unknown Crop")
        result = f"🌱 Recommended Crop: {crop}"

    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)