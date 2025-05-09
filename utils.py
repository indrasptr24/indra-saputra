from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

def load_hama_model():
    return load_model("model/model_hama.h5")

def predict_hama(img_path, model, class_names):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence
