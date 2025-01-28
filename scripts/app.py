from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import io

app = Flask(__name__)
model = load_model('model/model.h5')

# Cargar datos de las plantas
with open('plants_info.json', 'r') as f:
    plants_info = json.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    img_bytes = img_file.read()
    img = image.load_img(io.BytesIO(img_bytes), target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction[0])
    
    # Mapea el índice de la predicción a la clave correspondiente en el JSON
    class_keys = list(plants_info.keys())
    predicted_class = class_keys[predicted_class_index]

    # Obtener nombre y datos de la planta
    plant_name = plants_info[predicted_class]["name"]
    plant_info = plants_info[predicted_class]["info"]
    plant_bene = plants_info[predicted_class]["benefits"]

    return jsonify({'prediction': plant_name, 'info': plant_info, 'benefits': plant_bene})

if __name__ == '__main__':
    app.run(debug=True)
