import os
import json

def generate_plants_info(dataset_path):
    plants_info = {}
    class_dirs = os.listdir(dataset_path)

    for class_dir in class_dirs:
        if os.path.isdir(os.path.join(dataset_path, class_dir)):
            plant_name = class_dir.replace('_', ' ').title()
            plants_info[class_dir] = {
                "name": plant_name,
                "info": f"Información sobre {plant_name}.",
                "benefits": f"Beneficios de {plant_name}."
            }

    with open('plants_info.json', 'w') as json_file:
        json.dump(plants_info, json_file, indent=4)

# Ruta del directorio de tu dataset
dataset_path = 'dataset/val'  # O el directorio que estás utilizando
generate_plants_info(dataset_path)
