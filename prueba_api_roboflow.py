from inference_sdk import InferenceHTTPClient
from PIL import Image
import cv2
import numpy as np

# Configuración del cliente de Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="2GQYhlGQWADhgmIt7mJA"
)

# Función para probar la API con una imagen
def test_roboflow_api(image_path):
    try:
        # Abrir la imagen usando PIL
        image = Image.open(image_path).convert("RGB")

        # Realizar inferencia en la API
        result = CLIENT.infer(image, model_id="face-detection-mik1i/18")

        # Imprimir resultados
        print("Resultado de la API:")
        print(result)

        # Dibujar las detecciones si las hay
        if result and "predictions" in result and len(result["predictions"]) > 0:
            # Convertir la imagen de PIL a formato OpenCV (RGB -> BGR)
            image_np = np.array(image)
            image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            for detection in result["predictions"]:
                # Extraer coordenadas del centro, ancho y alto
                center_x, center_y = detection["x"], detection["y"]
                width, height = detection["width"], detection["height"]
                confidence = detection["confidence"]

                # Calcular esquinas del bounding box
                top_left_x = int(center_x - (width / 2))
                top_left_y = int(center_y - (height / 2))
                bottom_right_x = int(center_x + (width / 2))
                bottom_right_y = int(center_y + (height / 2))

                # Dibujar el bounding box original
                cv2.rectangle(
                    image_np_bgr,
                    (top_left_x, top_left_y),
                    (bottom_right_x, bottom_right_y),
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    image_np_bgr,
                    f"Conf: {confidence:.2%}",
                    (top_left_x, top_left_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

            # Mostrar la imagen anotada (BGR -> RGB para mostrar correctamente)
            image_np_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
            Image.fromarray(image_np_rgb).show()
        else:
            print("No se detectaron caras.")

    except Exception as e:
        print(f"Error al llamar a la API: {e}")


# Probar con una imagen de ejemplo
test_roboflow_api("src/images/jhon_cv(1).jpg")
