import tensorflow as tf

# Load your existing Keras model
model = tf.keras.models.load_model("saved_model/detect_model.keras")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("saved_model/detect_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model converted and saved as detect_model.tflite")
