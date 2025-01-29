import os
from tensorflow.keras.models import load_model

def generate_model_summary(model_path, summary_file):
    # Cargar el modelo
    model = load_model(model_path)
    
    # Redirigir la salida a un archivo
    with open(summary_file, 'w', encoding='utf-8') as f:
        # Crear una función para redirigir la salida de impresión al archivo
        def print_to_file(*args, **kwargs):
            print(*args, **kwargs, file=f)
        
        # Imprimir el resumen del modelo al archivo
        model.summary(print_fn=print_to_file)

# Ruta del modelo y archivo de resumen
model_path = 'model/model.h5'
summary_file = 'model/model_summary.txt'

# Generar el resumen del modelo
generate_model_summary(model_path, summary_file)

print(f"Resumen del modelo guardado en {summary_file}")
