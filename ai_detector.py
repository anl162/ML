import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = tf.keras.models.load_model("ai_detector_model.h5")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        return (f"{img_path} → AI-generated (Confidence: {prediction:.2f})")
    else:
        return (f"{img_path} → Real (Confidence: {1-prediction:.2f})")

# Example usage
if __name__ == "__main__":
    img_path = input("Enter image path: ")
    predict_image(img_path)
