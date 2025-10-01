from flask import Flask, render_template, request
import numpy as np
import cv2
import os
import tensorflow as tf

app = Flask(__name__)

# ✅ Load Keras model
MODEL_PATH = os.path.join("saved_model", "detect_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

# Labels
labels = {0: "Normal", 1: "Pneumonia"}

# Helper function to preprocess image and predict
def prepare_and_predict(img_path, img_size=150):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # grayscale
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)  # channel dim
    img = np.expand_dims(img, axis=0)   # batch dim

    prediction = model.predict(img, verbose=0)[0][0]
    return labels[int(np.round(prediction))]

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    prediction, img_path = None, None
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction="No file uploaded")
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", prediction="No file selected")
        
        # Save uploaded file
        img_path = os.path.join("static", file.filename)
        file.save(img_path)

        # Predict
        prediction = prepare_and_predict(img_path)

    return render_template("index.html", prediction=prediction, img_path=img_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
