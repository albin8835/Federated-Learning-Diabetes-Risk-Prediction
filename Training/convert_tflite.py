import tensorflow as tf

print("Loading Keras model...")

model = tf.keras.models.load_model("global_model.keras")

print("Converting to TensorFlow Lite (Float32)...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# IMPORTANT:
# Do NOT enable optimization

tflite_model = converter.convert()

with open("model_float32.tflite", "wb") as f:
    f.write(tflite_model)

print("Done!")
print("Saved as model_float32.tflite")