from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

# Crear la aplicación FastAPI
app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas las fuentes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ruta al modelo guardado
MODEL_PATH = "my_model.keras"  # Cambia esta ruta si el modelo está en otro lugar
model = tf.keras.models.load_model(MODEL_PATH)

# Clase para las predicciones
class_names = ['Gato', 'Perro']

def prepare_image(file: BytesIO):
    """
    Procesa la imagen subida para ajustarla al formato esperado por el modelo.
    """
    img = Image.open(file).convert("RGB")  # Abrir y convertir a RGB
    img = img.resize((150, 150))  # Cambiar tamaño
    img_array = np.array(img) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)  # Expandir dimensiones para el modelo
    return img_array

@app.get("/")
async def read_root():
    return {"message": "Bienvenido al Clasificador de Imágenes"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint para realizar predicciones con una imagen subida.
    """
    try:
        # Leer la imagen desde el archivo subido
        contents = await file.read()
        img_array = prepare_image(BytesIO(contents))  # Preparar la imagen
        prediction = model.predict(img_array)  # Predecir con el modelo
        confidence = float(prediction[0][0])  # Confiabilidad de la predicción
        class_name = class_names[int(confidence > 0.5)]  # Convertir a clase

        return {
            "prediction": class_name,
            "confidence": confidence  # Devuelve la confianza en formato decimal
        }
    except Exception as e:
        return {"error": str(e)}

# Ejecutar el servidor: `uvicorn servidor:app --reload --host 0.0.0.0 --port 5000`
