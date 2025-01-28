import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuración del generador de datos
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    'dataset/train',            # Directorio de entrenamiento
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

val_gen = datagen.flow_from_directory(
    'dataset/val',              # Directorio de validación
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Crear el modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(150, 150, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(train_gen.class_indices), activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_gen, validation_data=val_gen, epochs=10)

# Guardar el modelo
model.save('model/model.h5')
