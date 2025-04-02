from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
from pyngrok import ngrok  # Optional (for public URL)

app = Flask(__name__)

# Load model (ensure 'crop_recommendation_model.pkl' exists in the same folder)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'crop_recommendation_model.pkl')

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500

    try:
        # Extract the 4 original variables
        data = {
            'P': float(request.form['P']),
            'K': float(request.form['K']),
            'temperature': float(request.form['temperature']),
            'humidity': float(request.form['humidity'])
        }
        features = np.array([[data['P'], data['K'], data['temperature'], data['humidity']]])
        prediction = model.predict(features)[0]
        return jsonify({'status': 'success', 'prediction': prediction})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == "__main__":
    # Optional: Ngrok tunnel (comment out if unused)
    try:
        ngrok_tunnel = ngrok.connect(5000)
        print(" * Public URL:", ngrok_tunnel.public_url)
    except:
        print(" * Ngrok not configured. Running locally.")

    app.run(host='0.0.0.0', port=5000, debug=True)