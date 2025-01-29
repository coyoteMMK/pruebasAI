import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import io

# Cargar el modelo
model = load_model('model.h5')

# Cargar datos de las plantas
with open('plants_info.json', 'r') as f:
    plants_info = json.load(f)

# Función de predicción
def predict_plant(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
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
    plant_benefits = plants_info[predicted_class]["benefits"]

    return {'prediction': plant_name, 'info': plant_info, 'benefits': plant_benefits}

# Crear la interfaz Gradio
interface = gr.Interface(
    fn=predict_plant,
    inputs=gr.Image(type="filepath"),
    outputs=gr.JSON(),
    live=False
)

# Iniciar la interfaz
if __name__ == "__main__":
    interface.launch()