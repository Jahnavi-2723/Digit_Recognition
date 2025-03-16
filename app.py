
from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os

app = Flask(__name__)

# Load the trained model
model = load_model("Digit_Recognition.h5")

# Set upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("front1.html")  # Load the web UI

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    # Save the uploaded image
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Preprocess the image (modify this part)
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))  # Resize to match MNIST format
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)  # Invert if needed
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=-1)  # Ensure correct shape
    image = np.expand_dims(image, axis=0)  # Add batch dimension


    # Make prediction
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)

    return jsonify({"prediction": int(predicted_digit), "image_path": file_path})

if __name__ == "__main__":
    app.run(debug=True)
