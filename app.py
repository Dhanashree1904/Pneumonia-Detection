from flask import Flask, render_template, request
from tensorflow import keras
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load your saved Keras model
MODEL_PATH = os.path.join("saved_model", "detect_model.keras")
model = keras.models.load_model(MODEL_PATH)

# Labels
labels = {0: "Normal", 1: "Pneumonia"}

# Helper function to process image
def prepare_image(img_path, img_size=150):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # assuming your model is grayscale
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)  # add channel dimension
    img = np.expand_dims(img, axis=0)   # add batch dimension
    return img

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction="No file uploaded")
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", prediction="No file selected")
        
        # Save uploaded image temporarily
        img_path = os.path.join("static", file.filename)
        file.save(img_path)

        # Prepare image and predict
        img = prepare_image(img_path)
        pred = model.predict(img)
        class_idx = int(np.round(pred[0][0]))  # for binary classification
        prediction = labels[class_idx]

        # Optionally remove uploaded image after prediction
        # os.remove(img_path)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

