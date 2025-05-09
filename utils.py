from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown

def load_hama_model():
    model_path = "model/model_hama.h5"
    
    os.makedirs("model", exist_ok=True)
    url = "https://drive.google.com/uc?id=1ahp6x3MDsBLz_M0vcRi6uq64XYlNb_TL" 
    gdown.download(url, model_path, quiet=False)

    if not os.path.exists(model_path):
        print("File model tidak ditemukan setelah diunduh!")
    else:
        print("File model berhasil diunduh.")
    return load_model(model_path)

def predict_hama(img_path, model, class_names):
    img = image.load_img(img_path, target_size=(220, 220))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence
